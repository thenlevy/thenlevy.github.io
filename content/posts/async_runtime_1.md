---
date: "2026-04-19T11:15:00+02:00"
draft: false
title: async_runtime Part I: Introduction to asynchronous programming in Rust.
description: "Introduction to asynchronous programming and the tasks that an asynchronous runtime must perform."
---

Asynchronous programming is a topic that is discussed a lot in the Rust community, and is more generally gaining popularity as a way to perform computations in I/O-extensive environments.
In these series of posts, we will explore how Rust handles asynchronous computations by implementing our own asynchronous runtime.

# Asynchronous programming

_Concurrency_ is the ability for a program to make progress on several tasks over overlapping periods of time.
This can be achieved by either:

- Performing tasks simultaneously on several CPU cores, which is known as _parallelism_ or _multi-threading_
- Context switching between tasks, which we will refer to as _interleaving_ tasks.

In practice context-switching is almost always going to happen in our system as the number of processes running on a computer outnumbers the number of CPU cores available.
The real design choice is therefore not _whether_ tasks will be interleaved, but _how_ the scheduling of the tasks will be decided. There are therefore two options:

- **Delegating scheduling to the OS**: spawn one thread per task and let the kernel schedule them.
  Each thread appears to own a CPU; the OS preempts it at arbitrary points to give time to another thread.
  This allows the code to follow a linear execution flow, but comes with a few performance costs (kernel bookkeeping, a dedicated stack, context switches that cross into kernel space), and the switch points are outside the program's control.
- **Scheduling inside the program**: keep the pool of OS threads small and multiplex many tasks on top of them in user space.
  Context-switching is done within the thread at controlled points (typically when a task can no longer make progress until one of its dependencies is resolved).
  The switch is then no more expensive than a function call.
  However, the program must now _supply its own scheduler_, and _a task that forgets to yield will starve every other task sharing its thread._

The need for tasks to be able to yield control to the scheduler becomes critical once we look at what tasks actually do while they are _waiting_. Most real tasks depend on inputs they cannot produce themselves: a reply from a remote server, a chunk read from disk, a timer firing, or the result of another task. There are two fundamentally different ways to wait for such a dependency:

- **Blocking wait**: We block the current OS thread until the dependency is resolved, registering what the task is waiting for so that the OS can wake the thread when relevant.
  This way of waiting is natural when using OS-level scheduling, but cannot be used when scheduling is done in user-space context where OS threads are seen as a limited resource.
- **Non-blocking wait**: The task that can no longer make progress registers what it needs to resume its execution, and the thread switches back to the scheduler which selects another task to run.

Explicit management of interleaving tasks, together with the use of non-blocking waits, is what the term **Asynchronous programming** refers to.

## Asynchronous computations in Rust

In Rust, a task whose scheduling is managed by the program itself is represented by a value whose type implements the `Future` trait.

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

{{< callout type="note" icon="📌" title="About `Pin<&mut Self>`" >}}
`Pin<Ptr>` is a wrapper arround a type `Ptr` that is a pointer to some type `V` (i.e. `Ptr` implements `Deref/Mut<Traget = V>`).
Wrapping a pointer inside a `Pin` carries the guarantee that the value pointed verifies _at least one_ of the following properties:

- The bytes representing the value will not be re-loacted to a different memory address until the value (not the `Pin`-wrapped pointer!) is dropped.
- The type `V` implements the trait `Unpin`, which means that it is actually safe to move the value to a different memory address.

We will see later why these guarantees are important when we'll look at example of implementions of the `Future` trait, but in the context of understanding what the `poll` method does, we should simply see `self: Pin<&mut Self>` as "`&mut self` with some extra guarantees".
{{< /callout >}}

Where `Output` is the type of the value produced by the computation.

If `fut` is a value of type `Fut` that implements `Future<Output = T>`, then `fut` holds all the context necessary to perform the computation, as well as to store it's intermediate results before handing control back to the scheduler.

The `poll` function is to be called the scheduler. When this function is called, either:

- The computation can be run to completion and yields its result, in which case the `poll` function returns `Poll::Ready(result)`.
- The computation cannot be run to completion yet, in which case the `poll` function, will _ensure that it is rescheduled later_ by registering itself in the Context `cx`, and return `Poll::Pending`. During execution of `poll` the value of `fut` may be updated in whatever way is necessary to store the intermediate results and resume the computation later.

{{< callout type="note" icon="🤯" title="How do tasks know when to resume?" >}}
The exact mechanism by which tasks register what they are waiting for and ensure that they are resumed when they can make further progress is often not discussed in tutorials about asynchronous programming. Understanding how this mechanism works is what motivated me to dive into the details of asynchronous runtimes implementation.

In my opinion, understanding how this mechanism works requires to have an overview of the different components of an asynchronous runtime.
For this reason we will take it for granted for now and come back to it later when we understand better how runtimes work.
{{< /callout >}}

## `async/await`

While the `Future` trait can be implemented manually, it is mostly used as a "low-level" intefarce to implement the elementary building blocks of assynchronous functions.

The `async/await` syntax is a high-level interface for writing asynchronous code: it lets you mark the **suspension points**—the places where execution may pause because a dependency is not yet ready.

The `async` keyword can decorate a function, a block, or a closure.
Using `async` has two consequences:

- If the block would evaluate to a value of type `T`, or the function or closure would return `T`, then wrapping it in `async` yields a value of an opaque type that implements `Future<Output = T>`.
- Only inside `async` code may you use the `await` operator on a value `fut` whose type implements `Future<Output = T>`.

That opaque type is produced by the compiler, it can be used as an `impl Future<Output = T>` but cannot be referred to by name. This is similar to how the type of a closure can only be used as implementing a function trait.

This opaque type implements `Future<Output = T>`, and its fields _hold the inner state of the computation_: which part of the body should run next, locals that must survive across `await` points, and whatever handles the compiler needs to keep track of inner futures.
Conceptually, the value is a **state machine** whose `poll` advances one step at a time.

The compiler also emits the implementation of `poll` for that type. It corresponds to the code wapped inside `async`, with the additional logic to suspend and resume the computation at each `await`.
More precisely, the control flow is linear between two `await` points.
When an `await` point is reached, the generated code polls the awaited future and either continues if it is ready, or returns `Pending` and records how to pick up where it left off on the next poll.
