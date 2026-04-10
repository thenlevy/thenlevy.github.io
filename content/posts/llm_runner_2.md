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
It consists of a small JSON header followed by a blob that consists of the raw bytes for the model's parameters. The JSON header maps each tensor name to object containing information about the offset at which the tensor's parameters are stored in the blob following the header.
There are two resons for us to load the model from the `.safetensors` file instead of the `pytorch_model.bin` file:

- Compared to PyTorch, the Safetensors format is "safe" in the sense that unpacking a PyTorch file is a process that can lead to arbitrary code execution, meaning that it requires some trust in the source.
- Parsing is straightforawrd and there is pure Rust support for parsing the Safetensors format, using the [`safetensors`](https://docs.rs/safetensors/latest/safetensors/) crate.

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

When implementing the parsing, I thought the parameters of the vocab projector were missing from the safetensors file, so I've downloaded them in [Numpy format](https://numpy.org/doc/2.1/reference/generated/numpy.lib.format.html#format-version-1-0) from the pytorch model file using [Netron.app](https://netron.app/) and implemented parsing from this format

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

Later on, I learned that for the `distilbert-base-uncased` model, the parameters of the vocab projector are actually not made explicit in the safetensors file, but are instead the same as the one used in the embedding layer.

{{< /callout >}}

# Running the model on user input

The DistilBERT model that we've parsed here is an ecoder only model: it does not have a decoder stack so it does not produce a distribution over the _next_ token of a sequence.
Instead, it is trained to perform masked language modeling (MLM), that is to say to predict tokens that should replace occurences of a special `[MASK]` token in a sequence.

For this reason we will do not use it for a “keep appending tokens” inference here; we mimic a form of inference by filling a `[MASK]` slot.

Before feeding it to the model, we need to tokenize the input text. For that we will use the [tokenizers](https://docs.rs/tokenizers/latest/tokenizers/) crate, that we configure with the `tokenizer.json` file that comes with the model.

The example binary `examples/mlm_complete.rs` does the following:

1. Take a user prompt from the command line (or a default string).
2. Append `[MASK].` so there is a definite masked token to predict.
3. Resolve `[MASK]`’s id with `token_to_id("[MASK]")`.
4. Encode the string, collect `input_ids`, find the index of the mask id in that vector.
5. Load `model.safetensors`, build `DistilBert::try_from_bytes`, check `input_ids.len() <= distilbert.seq_len`.
6. Call `distilbert.evaluate(&input_ids)`, which returns a matrix of shape `(seq_len, vocab_size)` — the same projection head described in Part I, applied after the encoder and the DistilBERT MLM bottleneck.
7. Take `logits.row(mask_pos)`: a length-`vocab_size` vector of scores. Sort by logit and print the top-$k$ token ids and string pieces (via `id_to_token`).

So for prompt “The capital of France is” we might append `[MASK].` and inspect which tokens the model scores highest at the mask position—roughly “what word belongs here?” rather than “what is the next token in a continuation?”.

{{< callout type="note" icon="🤔" title="Why this is still instructive" >}}
It exercises the full path: tokenizer → ids → embeddings → transformer stack → MLM head → argmax or ranking over the vocabulary. That is the same numerical pipeline that powers masked pre-training, and it validates that our parsed weights and forward pass match the architecture we built in Part I.
{{< /callout >}}

# Summary

- We clone [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased) with Git (and Git LFS) and point the project at `model.safetensors` and `tokenizer.json`.
- Safetensors stores named tensors in a documented, safe binary format; the `safetensors` crate gives `SafeTensors::deserialize` and `TensorView` (`shape`, raw `data`), which `Matrix::try_from_view` and `Vector::try_from_view` turn into owned `f32` data for `Matrix`, `Vector`, `Norm`, and then `DistilBert`.
- The `tokenizers` crate loads `tokenizer.json`; we encode text with special tokens, then run `evaluate` and read MLM logits at the `[MASK]` position to simulate a concrete “inference” story for an encoder-only checkpoint.

Next steps—if we continue the series—might include causal decoders, sampling, or digging into how the tokenizer is built. For now, the runnable path is `cargo run --example mlm_complete -- "your words here"` with the cloned model directory in place, as sketched above.
