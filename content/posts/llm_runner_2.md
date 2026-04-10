---
date: "2026-04-10T08:00:00+02:00"
draft: false
title: "`llm_runner` Part II: Loading Weights and Running DistilBERT"
description: "Parsing Hugging Face checkpoints with safetensors, tokenizing text, and using masked language modeling to exercise our Rust encoder."
---

In [Part I]({{< relref "/posts/llm_runner_1" >}}), we implemented the DistilBERT architecture in Rust: embeddings, transformer blocks, and the vocabulary projection head. The tensors inside `DistilBert` were only *shapes* and *operations*; we still had to load real weights from a trained model and feed tokenized text into `evaluate`.

This post explains how we do that in [`llm-runner`](https://github.com/thenlevy/llm-runner): where the checkpoint comes from, what the [Safetensors](https://huggingface.co/docs/safetensors/index) format provides, how the [`safetensors`](https://docs.rs/safetensors/latest/safetensors/) crate exposes it, and how we wire that into `DistilBert::try_from_bytes`. We then run the model in a small example using the [tokenizers](https://docs.rs/tokenizers/latest/tokenizers/) crate and masked language modeling (MLM) so we can see logits over the vocabulary for a user-chosen prompt.

# Getting the model from Hugging Face

We target [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased), a compact encoder-only model trained with the same objectives as BERT (including MLM). Each model on the Hub is a normal Git repository; weights and tokenizer files are stored with [Git LFS](https://git-lfs.com/). Clone it with plain Git (after a one-time `git lfs install` on your machine):

```bash
git lfs install
git clone https://huggingface.co/distilbert-base-uncased distilbert-base-uncased
cd distilbert-base-uncased
git lfs pull
```

If `model.safetensors` is only a few hundred bytes after clone, LFS objects were not fetched yet—`git lfs pull` (or recloning with LFS set up first) fixes that.

For our code and tests we assume a directory next to the crate, e.g. `../distilbert-base-uncased/`, containing at least:

- `model.safetensors` — serialized weights
- `tokenizer.json` — tokenizer in the format produced by Hugging Face’s `tokenizers` library (this is what `Tokenizer::from_file` loads)

`config.json` documents hyperparameters (hidden size, number of layers, etc.); we mostly infer compatible dimensions from the tensors while parsing (for example, embedding layer norm gives us `d_model`).

# The Safetensors format

[Safetensors](https://huggingface.co/docs/safetensors/index) is a simple, mmap-friendly layout for storing a dictionary of named tensors: a small JSON header (tensor names, dtypes, shapes, and byte offsets) followed by the raw tensor bytes. It was designed as a safer alternative to unpickling arbitrary Python objects: you get typed arrays, not executable code.

The specification and rationale live in the [official documentation](https://huggingface.co/docs/safetensors/index) and the [`safetensors` repository](https://github.com/huggingface/safetensors). Many models on the Hub are published in this format alongside or instead of PyTorch `.bin` checkpoints.

# The `safetensors` crate: `SafeTensors` and `TensorView`

In Rust we depend on [`safetensors`](https://docs.rs/safetensors/latest/safetensors/) (see `Cargo.toml` in `llm-runner`). The typical entry point is:

1. `SafeTensors::deserialize(bytes)` parses the file in memory and returns a `SafeTensors` handle.
2. `tensor(name)` looks up a tensor by its full string key (as stored in the file, e.g. `distilbert.embeddings.word_embeddings.weight`) and returns a `TensorView<'_>` on success.

A [`TensorView`](https://docs.rs/safetensors/latest/safetensors/tensor/struct.TensorView.html) is a zero-copy view into the original buffer:

- `shape()` returns the dimensions (for matrices, two lengths; for vectors, one).
- `data()` exposes raw bytes for the tensor’s elements; for `f32` we interpret each four-byte little-endian chunk as one float.

The file’s header already recorded dtype and shape; `TensorView` is how we see that layout in Rust. The next step is to validate slices of that shape against what we have learned so far (for example, once `d_model` is known, we can require the second axis of `word_embeddings.weight` to match it while still reading the vocabulary size from the tensor’s first axis) and to copy into owned [`nalgebra`](https://docs.rs/nalgebra/latest/nalgebra/) types for `evaluate`. That is what `try_from_view` does: optional expected dimensions catch wrong keys early, and we require the payload length to match `rows * cols * 4` (or `len * 4` for a vector) so we never mis-read a truncated or wrongly typed blob.

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

At parse time we usually do not know `d_model`, `seq_len`, or `vocab_size` in advance: they are whatever the checkpoint stores. `Option<usize>` in `expected_shape` encodes that: `None` means “take this axis from the tensor in the file”; `Some(n)` means “it must equal a value we already inferred from an earlier tensor.” Later, attention and FFN blocks pass `d_model` into `try_from_views` so every weight lines up with that width.

Below, `src/distilbert/parse.rs` deserializes the file, declares `seq_len`, `d_model`, and `vocab_size` without initial values, then fills them while building the embedding submodule. The `path` stack mirrors Hugging Face’s dotted names (`distilbert.embeddings.LayerNorm.bias`, …); each `path.join(".")` is the tensor key inside the safetensors archive. After `LayerNorm`, we know `d_model`; position and word matrices use `[None, Some(d_model)]` so row counts become `seq_len` and `vocab_size`. The block ends with an `Embeddings { norm, positions, words }` value ready for `DistilBert`.

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

The MLM head does the same for the inner dimension of `vocab_transform`: infer `d_logit` from the weight matrix, then fix the bias length.

```rust
        let vocab_transform_weight =
            Matrix::try_from_view(vocab_transform_weight_view, [None, Some(d_model)])?;
        let d_logit = vocab_transform_weight.shape()[0];
        let vocab_transform_bias_view = safe_tensors.tensor("vocab_transform.bias")?;
        let vocab_transform_bias = Vector::try_from_view(vocab_transform_bias_view, Some(d_logit))?;
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

`Norm::try_from_views` in `src/layers/norm.rs` composes two such views (bias and weight) with matching length. Together, these are the bridge from Safetensors’ flat, named tensors to the typed matrices and vectors used in Part I.

# Parsing into `DistilBert`

Implementation lives in `src/distilbert/parse.rs`: `DistilBert::try_from_bytes` deserializes with `SafeTensors::deserialize`, then walks the Hugging Face / PyTorch parameter names for DistilBERT.

Rough structure:

1. `distilbert.embeddings` — layer norm (`LayerNorm.weight` / `bias`), `position_embeddings.weight`, `word_embeddings.weight` → our `Embeddings`. From the embedding norm we recover `d_model`, from position embeddings `seq_len`, from word embeddings `vocab_size`.
2. `distilbert.transformer.layer.{i}` — for each index `i` until `q_lin.weight` is missing, load attention (`q_lin`, `k_lin`, `v_lin`, `out_lin`), FFN (`ffn.lin1` / `lin2`), `sa_layer_norm`, and `output_layer_norm` → one `Stack` per layer.
3. Head for masked LM — keys at the top level of the state dict, e.g. `vocab_transform.weight`, `vocab_transform.bias`, `vocab_layer_norm`, `vocab_projector.bias`, assembled into our `VocabLayer`.

We build dot-separated names with a small path stack (e.g. `path.join(".")`) so the code mirrors the nested naming convention used on the Hub.

{{< callout type="note" icon="⚠️" title="Vocab projector weights" >}}
In the checkpoint used while developing `llm-runner`, the large `vocab_projector.weight` tensor was not present in `model.safetensors`. DistilBERT’s LM head can be tied to the input word embeddings in full implementations; depending on export tooling, that matrix may be omitted or stored elsewhere. Our parser currently loads that matrix from a separate `vocab_projector_weight.npy` file next to the model directory (bytes parsed after the NumPy header). If your `.safetensors` file includes `vocab_projector.weight`, you would map it the same way as the other matrices and drop the sidecar. The important idea for this post is that every slot in `DistilBert` must be filled from some consistent source matching the training checkpoint.
{{< /callout >}}

The constructor ends with something like:

```rust
Ok(Self {
    embeddings: embedding,
    encoder: transformers,
    d_model,
    seq_len,
    vocab_size,
    vocab_layer,
    n_heads: 12,
})
```

where `n_heads` matches `distilbert-base-uncased` (12 heads, `d_model = 768`). A unit test in `src/lib.rs` asserts `d_model`, `seq_len`, `vocab_size`, and layer count after parsing.

# Running the model on user input

DistilBERT as released on the Hub is not a causal language model: it does not produce a distribution over the *next* token given all previous tokens in an autoregressive loop. It is trained as an encoder with an MLM head: for each position, the network can output logits over the whole vocabulary, usually after seeing the full sequence (with some positions masked during training).

So we do not implement “keep appending tokens” inference here; we mimic a useful form of inference by filling a `[MASK]` slot.

## Tokenization with `tokenizers`

The [tokenizers](https://docs.rs/tokenizers/latest/tokenizers/) crate loads the same artifact Hugging Face ships with the model:

```rust
let tokenizer = Tokenizer::from_file(tokenizer_path)?;
```

`tokenizer.json` encodes the WordPiece (or other) model, special tokens, and normalization—no need to reimplement subword splitting in Rust.

For BERT-style models, a single sequence is encoded with special tokens `[CLS]` … `[SEP]`. In our example we call `encode(text, true)` so those tokens are added automatically. The result exposes token ids as `u32` values compatible with `DistilBert::evaluate`.

## MLM: one masked position, one row of logits

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
