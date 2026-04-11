---
date: "2026-04-10T08:00:00+02:00"
draft: false
title: "`llm_runner` Part II: Loading Weights and performing MLM inference"
description: "Loading weights from the DistilBERT model and performing MLM inference in llm-runner."
---

In [the previous post]({{< relref "/posts/llm_runner_1" >}}), we went through the mathematics and Rust implementation of an encoder architecture.
Now, let's see how we can download a model and parse it into our `DistilBert` struct.
Then, we will test our llm runner by performing masked language modeling (MLM) inference on a user-chosen prompt.

# Getting the model from Hugging Face

We target [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased), a compact encoder-only model trained with the same objectives as BERT (including MLM). Each model on the Hub is a normal Git repository; weights and tokenizer files are stored with [Git LFS](https://git-lfs.com/). Clone it with plain Git (after a one-time `git lfs install` on your machine):

```bash
git lfs install
git clone https://huggingface.co/distilbert-base-uncased distilbert-base-uncased
cd distilbert-base-uncased
git lfs pull
```

When running the tests at the end of this post, we will assume a directory next to the crate, e.g. `../distilbert-base-uncased/`, containing at least:

- `model.safetensors` — serialized weights
- `tokenizer.json` — tokenizer in the format produced by Hugging Face’s `tokenizers` library (this is what `Tokenizer::from_file` loads)

# Parsing the Safetensors model into the `DistilBert` struct

[Safetensors](https://github.com/huggingface/safetensors) is an alternative to the format more commonly used in PyTorch.
It consists of a small JSON header followed by a blob that consists of the raw bytes for the model's parameters. The JSON header maps each tensor name to an object containing information about the offset at which the tensor's parameters are stored in the blob following the header.
There are two reasons for us to load the model from the `.safetensors` file instead of the `pytorch_model.bin` file:

- Compared to PyTorch, the Safetensors format is "safe" in the sense that unpacking a PyTorch file is a process that can lead to arbitrary code execution, meaning that it requires some trust in the source.
- Parsing is straightforward and there is pure Rust support for parsing the Safetensors format, using the [`safetensors`](https://docs.rs/safetensors/latest/safetensors/) crate.

The [`safetensors crate`](https://docs.rs/safetensors/latest/safetensors/) exposes the following interface:

- `SafeTensors::deserialize(bytes: &[u8]) -> Result<Safetensors, Error>`
- `Safetensors::tensor(name: &str) -> Result<TensorView, Error>` that returns a `TensorView` that borrows the underlying buffer: `shape()` gives dimensions (two for matrices, one for vectors), and `data()` is the raw bytes—here each four little-endian bytes are one `f32`.

Our parsing will rely on this interface and the two helper functions below that load the parameters of a tensor into a `Matrix` or `Vector` struct.

`src/layers/matrix.rs`:

```rust
impl Matrix {
    pub fn try_from_view(
        view: TensorView<'_>,
        expected_shape: [Option<usize>; 2],
    ) -> Result<Self, Error> {
        let [rows, cols] = view.shape() else {
            return Err(Error::InconsistentShape);
        };

        if expected_shape[0].is_some_and(|r| *rows != r) {
            return Err(Error::InconsistentShape);
        }

        if expected_shape[1].is_some_and(|c| *cols != c) {
            return Err(Error::InconsistentShape);
        }

        if view.data().len() != rows * cols * 4 {
            return Err(Error::InvalidData);
        }

        Ok(Self {
            inner: DMatrix::from_row_iterator(
                *rows,
                *cols,
                view.data()
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])),
            ),
        })
    }

    // ...
}
```

`src/layers/vector.rs`:

```rust
impl Vector {
    pub fn try_from_view(
        view: TensorView<'_>,
        expected_length: Option<usize>,
    ) -> Result<Self, Error> {
        let [len] = view.shape() else {
            return Err(Error::InconsistentShape);
        };

        if expected_length.is_some_and(|l| l != *len) {
            return Err(Error::InconsistentShape);
        }

        if view.data().len() != len * 4 {
            return Err(Error::InvalidData);
        }

        Ok(Self {
            inner: DVector::from_iterator(
                *len,
                view.data()
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])),
            ),
        })
    }

    // ...
}
```

The parsing walks along the tensor names, recovers their parameters from the `SafeTensors` handle and loads them into the fields of the `DistilBert` struct.
Here is an extract of the parsing, showing how we parse the embedding layer.

`src/distilbert/parse.rs` (from `SafeTensors::deserialize` through the embedding struct):

```rust
        let safe_tensors = SafeTensors::deserialize(bytes)?;

        let mut path = vec!["distilbert"];
        let seq_len;
        let d_model;
        let vocab_size;

        let epsilon = 1e-12;

        let embedding;
        {
            path.push("embeddings");
            let norm;
            {
                path.push("LayerNorm");
                path.push("bias");
                let bias_view = safe_tensors.tensor(&path.join("."))?;
                path.pop();
                path.push("weight");
                let weight_view = safe_tensors.tensor(&path.join("."))?;
                path.pop();
                norm = Norm::try_from_views(bias_view, weight_view, epsilon)?;
                d_model = norm.shape();
                path.pop();
            }
            path.push("position_embeddings.weight");
            let positions = Matrix::try_from_view(
                safe_tensors.tensor(&path.join("."))?,
                [None, Some(d_model)],
            )?;
            seq_len = positions.shape()[0];
            path.pop();
            path.push("word_embeddings.weight");
            let words = Matrix::try_from_view(
                safe_tensors.tensor(&path.join("."))?,
                [None, Some(d_model)],
            )?;
            vocab_size = words.shape()[0];

            path.pop();
            embedding = Embeddings {
                norm,
                positions,
                words,
            };
            path.pop();
        }
```

{{< callout type="note" icon="🤔" title="Vocab projector weights" >}}

When implementing the parsing, I thought the parameters of the vocab projector were missing from the safetensors file, so I've downloaded them in [NumPy format](https://numpy.org/doc/2.1/reference/generated/numpy.lib.format.html#format-version-1-0) from the pytorch model file using [Netron.app](https://netron.app/) and implemented parsing from this format

```rust
        let vocab_project_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../distilbert-base-uncased/vocab_projector_weight.npy");
        let vocab_project_bytes = std::fs::read(vocab_project_path)?;
        let header_size =
            u16::from_le_bytes([vocab_project_bytes[8], vocab_project_bytes[9]]) as usize;

        let header_end = 10 + header_size;

        let vocab_project_weight = Matrix::try_from_bytes(
            vocab_project_bytes[header_end..(header_end + d_logit * vocab_size * 4)].as_ref(),
            [vocab_size, d_logit],
        )?;
```

Later on, I learned that for the `distilbert-base-uncased` model, the parameters of the vocab projector are actually not made explicit in the safetensors file, but are instead the same as those used in the embedding layer.

{{< /callout >}}

# Running the model on user input

The DistilBERT model that we've parsed here is an encoder-only model: it does not have a decoder stack so it does not produce a distribution over the _next_ token of a sequence.
Instead, it is trained to perform masked language modeling (MLM), that is to say to predict tokens that should replace occurrences of a special `[MASK]` token in a sequence.

For this reason we do not use it for a “keep appending tokens” inference here; we mimic a form of inference by filling a `[MASK]` slot.

Before feeding it to the model, we need to tokenize the input text. For that we will use the [tokenizers](https://docs.rs/tokenizers/latest/tokenizers/) crate, that we configure with the `tokenizer.json` file that comes with the model.

We write a program that does the following:

1. Take a user prompt from the command line (or a default string).
2. Append `[MASK].` so there is a definite masked token to predict.
3. Resolve `[MASK]`’s id with `token_to_id("[MASK]")`.
4. Encode the string, collect `input_ids`, find the index of the mask id in that vector.
5. Load `model.safetensors`, build `DistilBert::try_from_bytes`, check `input_ids.len() <= distilbert.seq_len`.
6. Call `distilbert.evaluate(&input_ids)`, which returns a matrix of shape `(seq_len, vocab_size)` — the same projection head described in Part I, applied after the encoder and the DistilBERT MLM bottleneck.
7. Take `logits.row(mask_pos)`: a length-`vocab_size` vector of scores. Sort by logit and print the top-$k$ token ids and string pieces (via `id_to_token`).

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let model_dir = root.join("../distilbert-base-uncased");
    let model_path = model_dir.join("model.safetensors");
    let tokenizer_path = model_dir.join("tokenizer.json");

    let user_prompt: String = std::env::args()
        .skip(1)
        .fold(String::new(), |acc, arg| acc + " " + arg.as_str());

    if user_prompt.is_empty() {
        return Err("No prompt provided".into());
    }

    let text = format!("{user_prompt} [MASK].");

    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| e.to_string())?;
    let mask_id = tokenizer
        .token_to_id("[MASK]")
        .ok_or("tokenizer has no [MASK] token")? as u32;

    // Here, the tokenizer will add the special tokens `[CLS]` and `[SEP]` at the beginning and end
    // of the text. These tokens indicate the beginning and end of the sequence to process to the
    // DistilBERT model.
    let encoding = tokenizer
        .encode(text.as_str(), true)
        .map_err(|e| e.to_string())?;
    let input_ids: Vec<u32> = encoding.get_ids().iter().map(|&id| id as u32).collect();

    let mask_pos = input_ids
        .iter()
        .position(|&id| id == mask_id)
        .ok_or("no [MASK] in tokenized input (check spelling / tokenizer)")?;

    let model_bytes = std::fs::read(model_path)?;
    let distilbert = DistilBert::try_from_bytes(&model_bytes).map_err(|e| format!("{e:?}"))?;

    if input_ids.len() > distilbert.seq_len {
        return Err(format!(
            "sequence length {} exceeds model max {}",
            input_ids.len(),
            distilbert.seq_len
        )
        .into());
    }

    let logits = distilbert
        .evaluate(&input_ids)
        .map_err(|e| format!("{e:?}"))?;
    let row = logits.row(mask_pos);

    let mut scored: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("Prompt + mask: {text}");
    println!("Token IDs: {input_ids:?}");
    println!("MLM at position {mask_pos} — top predictions:");
    const TOP: usize = 5;
    for (rank, (tid, logit)) in scored.iter().take(TOP).enumerate() {
        let piece = tokenizer
            .id_to_token(*tid as u32)
            .unwrap_or_else(|| "?".to_string());
        println!(
            "  {}. id={} logit={:.3} token={piece:?}",
            rank + 1,
            tid,
            logit
        );
    }

    Ok(())
}
```

# Results and next steps

Here are some examples of the results we get when running the program with different prompts.

```
sh-5.3$ cargo run --release --example mlm_complete -- "The biggest planet in the solar system is"
    Finished `release` profile [optimized] target(s) in 0.04s
     Running `target/release/examples/mlm_complete 'The biggest planet in the solar system is'`
Prompt + mask:  The biggest planet in the solar system is [MASK].
Token IDs: [101, 1996, 5221, 4774, 1999, 1996, 5943, 2291, 2003, 103, 1012, 102]
MLM at position 9 — top predictions:
  1. id=13035 logit=12.284 token="jupiter"
  2. id=11691 logit=12.256 token="venus"
  3. id=26930 logit=11.775 token="pluto"
  4. id=21167 logit=11.517 token="neptune"
  5. id=7733 logit=11.503 token="mars"
```

```
sh-5.3$ cargo run --release --example mlm_complete -- "The capital of France is"
    Finished `release` profile [optimized] target(s) in 0.05s
     Running `target/release/examples/mlm_complete 'The capital of France is'`
Prompt + mask:  The capital of France is [MASK].
Token IDs: [101, 1996, 3007, 1997, 2605, 2003, 103, 1012, 102]
MLM at position 6 — top predictions:
  1. id=16766 logit=12.166 token="marseille"
  2. id=25387 logit=11.711 token="nantes"
  3. id=17209 logit=11.684 token="toulouse"
  4. id=3000 logit=11.662 token="paris"
  5. id=10241 logit=11.551 token="lyon"
```

```
sh-5.3$ cargo run --release --example mlm_complete -- "The fastest animal on earth is the"
    Finished `release` profile [optimized] target(s) in 0.19s
     Running `target/release/examples/mlm_complete 'The fastest animal on earth is the'`
Prompt + mask:  The fastest animal on earth is the [MASK].
Token IDs: [101, 1996, 7915, 4111, 2006, 3011, 2003, 1996, 103, 1012, 102]
MLM at position 8 — top predictions:
  1. id=16490 logit=10.568 token="jaguar"
  2. id=10777 logit=9.546 token="elephant"
  3. id=7006 logit=8.891 token="lion"
  4. id=15450 logit=8.827 token="lizard"
  5. id=4419 logit=8.813 token="fox"
```

While these results are not perfect, we can at least see that the model is capable of generating a "reasonable" prediction for the masked token.
Considering the fact that these results are obtained with a small (64M parameters) model that can run on a laptop, I think this is rather encouraging, and it seems to indicate that the implementation is correct.

Here are things that I would like to explore in future posts:

- [Part III]({{< relref "/posts/llm_runner_3" >}}) covers parsing GPT-2 and text generation with a decoder ([PR #2](https://github.com/thenlevy/llm-runner/pull/2)).
- Looking at how the tokenizer is implemented
