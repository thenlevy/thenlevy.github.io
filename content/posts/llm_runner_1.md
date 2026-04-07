---
date: "2026-04-07T20:47:29+02:00"
draft: true
title: "`llm_runner` Part I: Loading and evaluating an Encoder"
description: "From Attention is all you need to a minimal Rust runner that loads weights and runs the encoder."
---

Recently, I've started a journey to "understand how LLM work".
Implementing something is a good way to understand it, so I've started the [`llm_runner`](https://github.com/thenlevy/llm_runner) project.

To start this project, it was natural to start with the architecture described in the fundamental paper _Attention is all you need_ [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762).

The goal is to implement a software capable of loading small models and evaluates their outputs on user-written propmt.

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

Here we will focus on the architecture described in [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762). This paper introduces the attention mechanism, which is the core of the Transformer architecture and is characterized by the use of this opperation:

$$\mathrm{Attention(Q, K, V)} = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where $Q$, $K$ and $V$ are matrices called the query, key and value matrices respectively. $d_k$ is the dimension of the key and value vectors. $\mathrm{softmax}$ is the softmax function.

$$\mathrm{softmax}(x) = \frac{\exp(x)}{\sum_{i=1}^{n}\exp(x_i)}$$

![The Transformer model architecture (Vaswani et al., 2017)](/assets/vas17_fig1.png)
_The Transformer model architecture as given in [[Vas17]](https://arxiv.org/abs/1706.03762)._

This schema describes the _architecture_ of a _Transformer_ model. It is composed of two stacks: an _encoder_ and a _decoder_.

The _encoder_ is responsible for encoding the input sequence into a sequence of hidden states. It can be seen as a function

$$\mathrm{encoder}: T^{\mathrm{seq\_len}} \rightarrow H^{\mathrm{seq\_len}}$$

Where $H$ is the _hidden state_ space. Vectors in $H$ embed a contextualized representation of the input sequence. This contextual representation is to be understood in the context of the attention mechanism, which is the core of the Transformer architecture and is described bellow.

The _decoder_ is responsible for generating the output sequence from the hidden states. It can be seen as a function

$$\mathrm{decoder}: H^{\mathrm{seq\_len}} \times T^{n_{\mathrm{out}}} \rightarrow [0, 1]^{n_T}$$

That will use the output of the encoder and the tokens of the output sequence so far to generate the next token in the sequence.
