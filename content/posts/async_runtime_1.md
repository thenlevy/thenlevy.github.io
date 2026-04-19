---
date: "2026-04-19T11:15:00+02:00"
draft: false
title: Building an assynchronous runtime in Rust
description: "Introduction to assynchronous programming and the tasks that an assynchronous runtime must perform."
---

Asynchronous programming is a topic that is discussed a lot in the Rust community, and is more generally gaining popularity as a way to perform computations in I/O-extensive environments.
In these series of posts, we will explore how Rust handles assynchronous computations by implementing our own assynchronous runtimes.

# Asynchronous programming

_Concurrency_ is the ability for a program to make progress on several tasks over overlapping periods of time.
This can be achieved by either:

- Performing tasks simultaneously on several CPU cores, which is known as _parralelism_ or _multi-threading_
- Context switching between tasks, which we will refer to as _interleaving_ tasks.

In practice context-switching is almost always going to happen in our system as the number of processes running on a computer outnumbers the number of CPU cores available.
The real design choice is therefore not _whether_ tasks will be interleaved, but _how_ the scheduling of the tasks will be decided. There are therefore two options:

- **Delegating scheduling to the OS**: spawn one thread per task and let the kernel schedule them.
  Each thread appears to own a CPU; the OS preempts it at arbitrary points to give time to another thread.
  This allows the code to follow a linear execution flow, but comes with a few performance costs (kernel bookkeeping, a dedicated stack, context switches that cross into kernel space), and make switch points are outside the program's control.
- **Scheduling inside the program**: keep the pool of OS threads small and multiplex many tasks on top of them in user space.
  Context-switching is done within the thread at controlled points (typically when a tasks can no longer make progress until one of its dependencies is resolved).
  The switch is then no more expensive than a function call.
  However, the program must now _supply its own scheduler_, and _a task that forgets to yield will starve every other task sharing its thread._

This last point becomes critical once we look at what tasks actually do while they are _waiting_. Most real tasks depend on inputs they cannot produce themselves: a reply from a remote server, a chunk read from disk, a timer firing, or the result of another task. There are two fundamentally different ways to wait for such a dependency:

- **Blocking wait**: We block the current OS thread of until the dependency is resolved, registering what the task is waitig for so that the OS can wake the thread when relevant.
  This way of waiting is natural when using OS-level scheduling, but cannot be used when scheduling is done in user-space context where OS threads are seen as a limited ressource.
- **Non-blocking wait**: The task that can no longer make progress registers what it needs to resume its execution, and the threads switches back to the scheduler which selects another task to run.

Explicit management of interleaving tasks and the use of non-blocking waits is what the term **Asynchronous programming** refers to.

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
