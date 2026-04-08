---
date: "2026-04-07T20:47:29+02:00"
draft: true
title: "`llm_runner` Part I: Loading and evaluating an Encoder"
description: "From Attention is all you need to a minimal Rust runner that loads weights and runs the encoder."
---

Recently, I've started a journey to "understand how LLM work".
Implementing something is a good way to understand it, so I've started the [`llm_runner`](https://github.com/thenlevy/llm_runner) project.

To start this project, it was natural to start with the architecture described in the fundamental paper _Attention is all you need_ [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762).

In this post, we will focus on the general mathematical operations performed by LLMs, and their implementation in Rust.

Later, we'll parse an actual model and make a programm that runs it on user-written prompts.

# What is a LLM?

LLMs can be seen as mathematical functions that transform sequences of tokens into a probability distribution over the token space.
That is to say a function
$$\mathrm{llm}: T^{\mathrm{seq\_len}} \rightarrow [0, 1]^{n_T}$$

where $T$ is the token space, $\mathrm{seq\_len}$ is the length of the sequence and $n_T$ is the size of the token space.

In Rust this would be written like this

```rust
fn llm(tokens: &[Token]) -> Vec<f32> {
    // ...
}
```

Given a sequence of tokens, an LLM can be used iteratively to generate the next token in the sequence, in a process called "Inference".

This would for example look like

```rust
fn infer(tokens: &mut Vec<Token>, n: usize) {
    for _ in 0..n {
        let output = llm(token.as_slice());

        // Alternatively, sample output as a probability distribution.
        let argmax = output.iter().enumerate().max_by_key(|(_, &x)| x).unwrap().0;
        tokens.push(Token::from_token_id(argmax));
    }
}
```

# How is the output of the LLM defined?

The mathematical operations performed by the LLM are determined by two components:

- The _architecture_ of the LLM, which defines the operations that are successively applied to the input.
- The _weights_ of the LLM, which are the parameters of these mathematical operations.

Here we will focus on the architecture described in [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762). This paper introduces the Transformer architecture, which leverage the attention mechanism, that is to say the use of this opperation:

$$\mathrm{Attention(Q, K, V)} = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where $Q$, $K$ and $V$ are matrices called the _query_, _key_ and _value_ matrices respectively. $d_k$ is the dimension of the key and value vectors. $\mathrm{softmax}$ is the softmax function.

$$\mathrm{softmax}(x) = \frac{\exp(x)}{\sum_{i=1}^{n}\exp(x_i)}$$

```rust
fn softmax(x: &[f32]) -> Vec<f32> {
    // Directly tranlating the above forumula would lead to computing exponential of large
    // numbers, leading to numeric instability (esp on `f32`s).
    // This implementation is mathematically equivalent, but more stable.
    let max = x.iter().max().unwrap();
    let sum = x.iter().map(|x| (x - max).exp()).sum();
    x.iter().map(|x| exp(x - max) / sum).collect()
}
```

![The Transformer model architecture (Vaswani et al., 2017)](/assets/vas17_fig1.png)
_The Transformer model architecture as given in [[Vas17]](https://arxiv.org/abs/1706.03762)._

This schema describes the _architecture_ of a _Transformer_ model. It is composed of two stacks: an _encoder_ and a _decoder_.

The _encoder_ is responsible for encoding the input sequence into a sequence of hidden states. It can be seen as a function

$$\mathrm{encoder}: T^{\mathrm{seq\_len}} \rightarrow H^{\mathrm{seq\_len}}$$

Where $H$ is the _hidden state_ space. Vectors in $H$ embed a contextualized representation of the input sequence. This contextual representation is to be understood in the context of the attention mechanism, which is the core of the Transformer architecture and is described bellow.

The _decoder_ is responsible for generating the output sequence from the hidden states. It can be seen as a function

$$\mathrm{decoder}: H^{\mathrm{seq\_len}} \times T^{n_{\mathrm{out}}} \rightarrow [0, 1]^{n_T}$$

That will use the output of the encoder and the tokens of the output sequence so far to generate the next token in the sequence.

In reality, the element of the input and output sequences do not live in the space of tokens, but in a space of _embeddings_ $E=\mathbb{R}^{n_E \times n_E}$. Where $n_E$ is smaller than $n_T$.

This means that the input is embedded in $E$ before being passed to the encoder. This embedding is defined as

$$\mathrm{embedding}: \{0, 1\}^{\mathrm{seq\_len}} \rightarrow E$$
$$\mathrm{embedding}(x) = \mathrm{LayerNorm}(xW_e + P_e) $$

Where $W_e \in \mathbb{R}^{\mathrm{seq\_len} \times n_E}$ is the _learnt_ matrix of embeddings and $P_e\in \mathbb{R}^{\mathrm{seq\_len} \times n_E}$ is a constant matrix of _positional encodings_, whose role is to encode information about the position of the tokens in the sequence within the embedding.

And $\mathrm{LayerNorm}$ is a layer normalization operation defined [here](<https://en.wikipedia.org/wiki/Normalization_(machine_learning)#Layer_normalization>) (see also the rust code bellow for definition).

```rust
// Matrix is a wrapper around nalgebra::Dmatrix<f32> with parsing convenience methods.

pub struct Embeddings {
    pub norm: Norm,
    /// Constant parameters, not updated during training.
    pub positions: Matrix,
    /// Learnt parameters, updated during training.
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

        self.norm.normalize_rows(&mut embeddings)?;

        Ok(embeddings)
    }
}

// Vector is a wrapper around nalgebra::DVector<f32> with parsing convenience methods.

pub struct Norm {
    /// Learnt parameters unique to each layer, updated during training.
    bias: Vector,
    /// Learnt parameters unique to each layer, updated during training.
    weight: Vector,
    /// Constant parameter accross the whole Model, not updated during training.
    espilon: f32,
}

impl Norm {
    pub fn shape(&self) -> usize {
        self.bias.len()
    }

    pub fn normalize_row(&self, row: &mut DVectorViewMut<f32>) -> Result<(), Error> {
        let n = row.len();
        if n != self.shape() {
            return Err(Error::InconsistentShape);
        }

        let mean = row.mean();
        let inv_std = 1.0 / (row.variance() + self.espilon).sqrt();

        *row -= &DVector::from_element(n, mean);

        *row *= inv_std;

        row.component_mul_assign(&self.weight);

        *row += &*self.bias;

        Ok(())
    }

    pub fn normalize_rows(&self, rows: &mut DMatrix<f32>) -> Result<(), Error> {
        let (_, n_cols) = rows.shape();

        if n_cols != self.shape() {
            return Err(Error::InconsistentShape);
        }

        rows.row_iter_mut().try_for_each(|mut row| {
            self.normalize_row(&mut row.as_view_mut())?;
            Ok(())
        })
    }
}
```
