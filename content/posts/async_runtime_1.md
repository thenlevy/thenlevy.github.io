---
date: "2026-04-19T11:15:00+02:00"
draft: false
title: Building an assynchronous runtime in Rust
description: "Introduction to assynchronous programming and the tasks that an assynchronous runtime must perform."
---

Asynchronous programming is a topic that is discussed a lot in the Rust community, and is more generally gaining popularity as a way to perform computations in I/O-extensive environments.
In these series of posts, we will explore how Rust handles assynchronous computations by implementing our own assynchronous runtimes.

# Assynchronous programming

_Concurrency_ is the ability for a program to have several tasks that are making progress at the same time, this can be achieved by:

- Simultaneous execution of multiple tasks, also known as _parallelism_ or _multi-threading_.
- Context switching between unfinished tasks.

It is a requirement for many computer programs to be able to perform several different tasks concurrently. For example, a web server may be responsible for serving requests from multiple clients concurrently.
Because the number of tasks to be performed concurrently often outnumbers the number of CPU cores available, some form of context switching is necessary most of the time. This context switching can either be achieved by:

- Spawning several threads to assign the tasks to these threads, delegating the scheduling of the tasks to the operating system.
- Handling scheduling of the tasks within the program itself.

Moreover, it is often the case that the computation power of a process is not the limiting factor for the completion of a task. This is the case when a task requires the output of other tasks, or external data to be completed.

In Rust, assynchronous computations are represented by values whose type implements the `Future` trait.

```rust
pub trait Future {
    type Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output>;
}

pub enum Poll<T> {
    Ready(T),
    Pending,
}
```

Where `Output` is the type of the value produced by the computation.
If `fut` is a value of type `Fut` that implements `Future<Output = T>`, "polling" `fut` will

- `Pin<Ptr>` is a wrapper arround a type `Ptr` that is a pointer to some type `V` (i.e. `Ptr` implements `Deref/Mut<Traget = V>`).
  Wrapping a pointer inside a `Pin` carries the guarantee that the value pointed verifies _ at least one_ of the following properties:

* The bytes representing the value will not be re-loacted to a different memory address until the value (not the `Pin`-wrapped pointer!) is dropped.
* The type `V` implements the trait `Unpin`, which means that it is actually safe to move the value to a different memory address.

We will see later why these guarantees are important when we'll look at example of implementions of the `Future` trait, but in the context of understanding what the `poll` method does, we should simply see `self: Pin<&mut Self>` as "`&mut self` with some extra guarantees".

- `Context<'_>` is a context object that is used to pass additional information to the `poll` method, as of April 2026, it is only used to pass a `Waker` to the `poll` method.

- `Poll<Output>` is an enum that represents the result of the `poll` method. It is defined as follows:

```rust
pub enum Poll<T> {
    Ready(T),
    Pending,
}
```
