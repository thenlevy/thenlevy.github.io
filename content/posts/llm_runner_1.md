---
date: "2026-04-07T20:47:29+02:00"
draft: true
title: "Building an LLM Runner in Rust."
---

Recently, I've started a journey to "understand how LLM work".
Implementing something is a good way to understand it, so I've started the [`llm_runner`](https://github.com/thenlevy/llm_runner) project.

To start this project, it was natural to start with the architecture described in the fundamental paper [Attention is all you need](https://arxiv.org/abs/1706.03762).

The goal is to implement a software capable of loading small models and evaluates their outputs on user-written propmt.

# What is a LLM?

LLMs can be seen as mathematical functions that transform sequences of tokens into a probability distribution over the token space.
That is to say a function
$$\mathrm{llm}: T^{\mathrm{seq\_len}} \rightarrow [0, 1]^{n_T}$$

where $T$ is the token space, $\mathrm{seq\_len}$ is the length of the sequence and $n_T$ is the size of the token space.
