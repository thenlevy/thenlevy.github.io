---
date: "2026-04-11T12:00:00+02:00"
draft: false
title: "`llm_runner` Part III: Parsing and evaluating GPT-2"
description: "Parsing and evaluating GPT-2 weights and running causal self-attention inference."
---

In [Part I]({{< relref "/posts/llm_runner_1" >}}) we implemented a Transformer encoder in Rust; in [Part II]({{< relref "/posts/llm_runner_2" >}}) we loaded DistilBERT and ran masked language modeling. This post describes the next step in [`llm-runner`](https://github.com/thenlevy/llm-runner): parsing **GPT-2** weights (targeting [`gpt2-medium`](https://huggingface.co/gpt2-medium)), evaluating the model with causal self-attention, and testing the pipeline.

This post accompanies [PR #2 on the `llm-runner` repository](https://github.com/thenlevy/llm-runner/pull/2).

# Adjustments to the Rust structures for GPT

## Optional normalization in embeddings

DistilBERT’s embedding module includes a `LayerNorm` after summing token and position embeddings. GPT-2 does not normalize that sum before the first block.

To reuse one `Embeddings` type for both models, we simply made the norm optional:

```10:38:src/layers/embeddings.rs
pub struct Embeddings {
    pub norm: Option<Norm>,
    pub positions: Matrix,
    pub words: Matrix,
}

impl Embeddings {
    pub fn embed(&self, input: &[u32]) -> Result<DMatrix<f32>, Error> {
        let [vocab_size, d_model] = self.words.shape();

        let mut embeddings = DMatrix::zeros(input.len(), d_model);

        for (i, token) in input.iter().enumerate() {
            let t_id = *token as usize;
            if t_id >= vocab_size {
                return Err(Error::InconsistentShape);
            }

            embeddings
                .row_mut(i)
                .copy_from(&(self.words.row(t_id) + self.positions.row(i)));
        }

        if let Some(norm) = &self.norm {
            norm.normalize_rows(&mut embeddings)?;
        }

        Ok(embeddings)
    }
}
```

## Conv1D vs PyTorch `Linear`

GPT-2 uses **Conv1D** to implement the modules for the Attention and FFN layers. This is a **feature-wise** linear map: for each sequence position, $y = x W + b$ with $W$ stored with a shape of `[in_features, out_features]`.

In contrast, the `linear` helper follows the **PyTorch `nn.Linear`** convention used for DistilBERT:
rows of $x$ are batch/sequence positions, and we multiply by $W^T$ so that the stored `Matrix` matches `[out_features, in_features]` after the same layout choices as in [Part II]({{< relref "/posts/llm_runner_2" >}}).

Moreover, in the GPT-2 checkpoint that we are using, the weights for the Attention layers are stored in a "packed" format where all the weights for the Q, K, and V matrices are concatenated together in a single matrix of shape `[d_model, 3 * d_model]`.

So in order to parse GPT-2 weights, we need to implement dedicated constructors for `Attention` and `Ffn`:

- **`Attention::try_from_fused_views`** loads the fused `c_attn` matrix `[d_model, 3 * d_model]`, splits it into three `[d_model, d_model]` blocks for Q, K, and V, and **transposes** each block so that subsequent `linear` calls match HF’s `x @ W` Conv1D math (`src/layers/attention.rs`).
- **`Ffn::try_from_transposed_views`** loads `mlp.c_fc` and `mlp.c_proj` in Conv1D shape and transposes them into the `linear_1` / `linear_2` layout expected by `Ffn::forward` (`src/layers/ffn.rs`).

```rust
    pub fn try_from_fused_views(
        views: FusedAttentionViews<'_>,
        d_model: usize,
    ) -> Result<Self, Error> {
        let fused = Matrix::try_from_view(views.c_attn_weight, [Some(d_model), Some(3 * d_model)])?;

        let bias_bytes = views.c_attn_bias.data();
        if bias_bytes.len() != 3 * d_model * 4 {
            return Err(Error::InvalidData);
        }

        // The weights W_q, W_k, W_v are stored in a single W_fused matix so that
        // [K, V, Q] = x @ W_fused.
        // Then, in GPT-2 implementation the attention scores are computed "Conv1D-style":
        // y = x @ W + b.
        // Since in our implementation we use a PyTorch-style linear layer
        //(y = x * Wᵀ + b), we need to transpose the weights here for consistency.
        let q_weights = {
            let block = fused.view((0, 0), (d_model, d_model));
            Matrix::from_dmatrix(block.transpose())
        };
        let k_weights = {
            let block = fused.view((0, d_model), (d_model, d_model));
            Matrix::from_dmatrix(block.transpose())
        };
        let v_weights = {
            let block = fused.view((0, 2 * d_model), (d_model, d_model));
            Matrix::from_dmatrix(block.transpose())
        };

        let q_bias = Vector::try_from_f32_le_bytes(&bias_bytes[0..d_model * 4], d_model)?;
        let k_bias =
            Vector::try_from_f32_le_bytes(&bias_bytes[d_model * 4..2 * d_model * 4], d_model)?;
        let v_bias =
            Vector::try_from_f32_le_bytes(&bias_bytes[2 * d_model * 4..3 * d_model * 4], d_model)?;

        let c_proj = Matrix::try_from_view(views.c_proj_weight, [Some(d_model), Some(d_model)])?;

        Ok(Self {
            q_weights,
            k_weights,
            v_weights,
            q_bias,
            k_bias,
            v_bias,
            out_weights: c_proj.transposed(),
            out_bias: Vector::try_from_view(views.c_proj_bias, Some(d_model))?,
        })
    }
```

```rust
    pub fn try_from_transposed_views(views: FfnViews, d_model: usize) -> Result<Self, Error> {
        let linear_1_raw = Matrix::try_from_view(views.linear_1, [Some(d_model), None])?;
        let hidden = linear_1_raw.shape()[1];
        let linear_1 = linear_1_raw.transposed();

        let linear_2_raw = Matrix::try_from_view(views.linear_2, [Some(hidden), Some(d_model)])?;
        let linear_2 = linear_2_raw.transposed();

        Ok(Self {
            linear_1,
            linear_2,
            bias_1: Vector::try_from_view(views.bias_1, Some(hidden))?,
            bias_2: Vector::try_from_view(views.bias_2, Some(d_model))?,
        })
    }
```

# Evaluation

## Masked (causal) self-attention

One of the key differences between the encoder and decoder stacks in the Transformer architecture as described in [[Vas17]](https://arxiv.org/abs/1706.03762) is how they use self-attention.

- In the encoder stack, the attention is **bidirectional**: each position attends to all other positions in the sequence.
- In the decoder stack, the attention is **causal**: each position attends only to previous positions in the sequence.

This is because the two stacks play different roles in the model:

- The encoder stack must encode a full understanding of the input sequence, and therefore needs to consider the context of all tokens at once.
- The decoder stack generates the output sequence one token at a time, so it must not attend to positions that correspond to non-generated tokens. This matters especially during training, when the model sees the full sequence and must not "cheat" by looking ahead at the token it is supposed to predict.

Causal attention is implemented by adding an **upper-triangular** block of $-\infty$ to attention scores before applying the row-wise softmax, so position $i$ only attends to $j \le i$.

```rust
    /// Equivalent to [forward_multi_head_masked] with a causal (upper triangular of -∞) mask.
    pub fn forward_multi_head_causal(
        &self,
        x: DMatrix<f32>,
        n_heads: usize,
    ) -> Result<DMatrix<f32>, Error> {
        self.forward_multi_head_impl(x, n_heads, AttnMask::Causal)
    }
```

```rust
    fn forward_multi_head_impl(
        &self,
        x: DMatrix<f32>,
        n_heads: usize,
        mask: AttnMask<'_>,
    ) -> Result<DMatrix<f32>, Error> {
        //...
        // ...
        for h in 0..n_heads {
            let c0 = h * d_head;
            let qh = q.view((0, c0), (seq, d_head));
            let kh = k.view((0, c0), (seq, d_head));
            let vh = v.view((0, c0), (seq, d_head));

            let mut scores = &qh * &kh.transpose();
            scores.scale_mut(scale);

            // Using -∞ as mask value does not break softmax because `f32::NEG_INFINITY.exp()`
            // returns 0.
            match &mask {
                AttnMask::None => {}
                AttnMask::Additive(m) => {
                    for i in 0..seq {
                        for j in 0..seq {
                            scores[(i, j)] += m[(i, j)];
                        }
                    }
                }
                AttnMask::Causal => {
                    for i in 0..seq {
                        for j in (i + 1)..seq {
                            scores[(i, j)] = f32::NEG_INFINITY;
                        }
                    }
                }
            }

            softmax_rows(&mut scores);

            let ctx = &scores * &vh;
            attended.view_mut((0, c0), (seq, d_head)).copy_from(&ctx);
        }
        // ...
    }
```

## Evaluating the GPT-2 model

GPT-2 is a decoder-only model: it forwards its input through several transformer blocks (each with attention and an FFN) and finally projects the last hidden state to vocabulary size to produce logits for the next token.

```rust
impl Gpt2 {
    pub fn evaluate(&self, input: &[u32]) -> Result<DMatrix<f32>, Error> {
        let mut x = self.embeddings.embed(input)?;

        for block in &self.blocks {
            let residual = x.clone();
            block.ln_1.normalize_rows(&mut x)?;
            let attn_out = block.attention.forward_multi_head_causal(x, self.n_heads)?;
            x = attn_out + residual;

            let residual = x.clone();
            block.ln_2.normalize_rows(&mut x)?;
            let mlp_out = block.mlp.forward(x)?;
            x = mlp_out + residual;
        }

        self.ln_f.normalize_rows(&mut x)?;

        let logits = &x * self.embeddings.words.transpose();
        Ok(logits)
    }
}
```

# Parsing and testing

Parsing reuses the same strategy as in [Part II]({{< relref "/posts/llm_runner_2" >}}): we call `SafeTensors::deserialize` to load the model, then map tensors into the `Gpt2` structure.

We add an example binary that loads the model and runs causal LM inference on a user-provided prompt.
As in [Part II]({{< relref "/posts/llm_runner_2" >}}), we use the [`tokenizers`](https://docs.rs/tokenizers/latest/tokenizers/) crate to tokenize the prompt, then call `evaluate` to obtain logits for the next token. Repeating that step extends the sequence token by token.

```rust
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| e.to_string())?;
    let encoding = tokenizer
        .encode(user_prompt.as_str(), false)
        .map_err(|e| e.to_string())?;
    let mut ids: Vec<u32> = encoding.get_ids().iter().map(|&id| id as u32).collect();

    let model_bytes = std::fs::read(&model_path)?;
    let gpt2 = Gpt2::try_from_bytes(&model_bytes).map_err(|e| format!("{e:?}"))?;

    print!("{user_prompt}");
    std::io::stdout().flush()?;

    for _ in 0..max_new_tokens {
        if ids.len() >= gpt2.seq_len {
            return Err(format!(
                "sequence length {} reached model max {}",
                ids.len(),
                gpt2.seq_len
            )
            .into());
        }

        let logits = gpt2.evaluate(&ids).map_err(|e| format!("{e:?}"))?;
        let next_id = sample_last(&logits, &mut rng);
        ids.push(next_id);

        let piece = tokenizer
            .decode(&[next_id], true)
            .map_err(|e| e.to_string())?;
        print!("{piece}");
        std::io::stdout().flush()?;
    }
```

Below are sample runs with different prompts.
The seed is passed in or printed before generation so runs stay reproducible.

```
±SEED=42 cargo run --release --example gpt2_complete -- "Hello, I am a software engineer"
    Finished `release` profile [optimized] target(s) in 0.04s
     Running `target/release/examples/gpt2_complete 'Hello, I am a software engineer'`
Hello, I am a software engineer. I work for enthusiasm on LINUX Culture, which is a typical business convention for people who is placing lengthy orders for sensor families for their machines. Please don
```

```
±cargo run --release --example gpt2_complete -- "We are going on vacation"
    Finished `release` profile [optimized] target(s) in 0.14s
     Running `target/release/examples/gpt2_complete 'We are going on vacation'`
seed: 7715690566371802064
We are going on vacation to the beach, yes I know climate change is not in the news right now, but I am getting older so this is where we are headed! We must
```

The continuations are only **32 new tokens**, so lines end mid-sentence (`Please don`, `We must`). Still, they show that the full path is working: tokenization, forward passes, and decoding back to text.
The output's quality is not perfect, grammar mistake and even incoherent word/sentences appear, but still, the output gives a vague impression of pseudo-coherence.

On my laptop, the program took about 57 seconds to generate 32 tokens. Speeding up evaluation could be a good topic for a later post. I am also curious to dig deeper into tokenization and eventually implement it ourselves.
