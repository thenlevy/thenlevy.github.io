---
date: "2026-04-19T11:15:00+02:00"
draft: false
title: "async_runtime Part I: Introduction to asynchronous programming in Rust."
description: "Introduction to asynchronous programming and the tasks that an asynchronous runtime must perform."
---

Asynchronous programming is a topic that is discussed a lot in the Rust community, and is more generally gaining popularity as a way to perform computations in I/O-extensive environments.
In this series of posts, we will explore how Rust handles asynchronous computations by implementing our own asynchronous runtime.

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
`Pin<Ptr>` is a wrapper around a type `Ptr` that is a pointer to some type `V` (i.e. `Ptr` implements `DerefMut<Target = V>`).
Wrapping a pointer inside a `Pin` carries the guarantee that the value pointed verifies _at least one_ of the following properties:

- The bytes representing the value will not be relocated to a different memory address until the value (not the `Pin`-wrapped pointer!) is dropped.
- The type `V` implements the trait `Unpin`, which means that it is actually safe to move the value to a different memory address.

We will see later why these guarantees are important when we'll look at examples of implementations of the `Future` trait, but in the context of understanding what the `poll` method does, we should simply see `self: Pin<&mut Self>` as "`&mut self` with some extra guarantees".
{{< /callout >}}

Where `Output` is the type of the value produced by the computation.

If `fut` is a value of type `Fut` that implements `Future<Output = T>`, then `fut` holds all the context necessary to perform the computation, as well as to store its intermediate results before handing control back to the scheduler.

The `poll` function is to be called by the scheduler. When this function is called, either:

- The computation can be run to completion and yields its result, in which case the `poll` function returns `Poll::Ready(result)`.
- The computation cannot be run to completion yet, in which case the `poll` function will _ensure that it is rescheduled later_ by registering itself in the Context `cx`, and return `Poll::Pending`. During execution of `poll` the value of `fut` may be updated in whatever way is necessary to store the intermediate results and resume the computation later.

{{< callout type="note" icon="🤯" title="How do tasks know when to resume?" >}}
The exact mechanism by which tasks register what they are waiting for and ensure that they are resumed when they can make further progress is often not discussed in tutorials about asynchronous programming. Understanding how this mechanism works is what motivated me to dive into the details of asynchronous runtime implementations.

In my opinion, understanding how this mechanism works requires having an overview of the different components of an asynchronous runtime.
For this reason we will take it for granted for now and come back to it later when we understand better how runtimes work.
{{< /callout >}}

## `async/await`

While the `Future` trait can be implemented manually, it is mostly used as a "low-level" interface to implement the elementary building blocks of asynchronous functions.

The `async/await` syntax is a high-level interface for writing asynchronous code: it lets you mark the **suspension points**—the places where execution may pause because a dependency is not yet ready.

The `async` keyword can decorate a function, a block, or a closure.
Using `async` has two consequences:

- If the block would evaluate to a value of type `T`, or the function or closure would return `T`, then wrapping it in `async` yields a value of an opaque type that implements `Future<Output = T>`.
- Only inside `async` code may you use the `await` operator on a value `fut` whose type implements `Future<Output = T>`.

That opaque type is produced by the compiler, it can be used as an `impl Future<Output = T>` but cannot be referred to by name. This is similar to how the type of a closure can only be used as implementing a function trait.

This opaque type implements `Future<Output = T>`, and its fields _hold the inner state of the computation_: which part of the body should run next, locals that must survive across `await` points, and whatever handles the compiler needs to keep track of inner futures.
Conceptually, the value is a **state machine** whose `poll` advances one step at a time.

The compiler also emits the implementation of `poll` for that type. It corresponds to the code wrapped inside `async`, with the additional logic to suspend and resume the computation at each `await`.
More precisely, the control flow is linear between two `await` points.
When an `await` point is reached, the generated code polls the awaited future and either continues if it is ready, or returns `Pending` and records how to pick up where it left off on the next poll.

## Example: A single threaded asynchronous TCP server

To illustrate the concepts we've seen so far, we will write a small program that implements a TCP server that can handle multiple clients concurrently on a single thread.

For now we will use the [`smol`](https://github.com/smol-rs/smol) runtime to drive our asynchronous computations, later in this series we will implement our own runtime and successively add the required functionalities to run a program with the same capabilities without our own runtime.

Our server:

- Accepts incoming connections on a given port.
- Plays a Guessing Game with all the clients: Clients have to find a hidden number and the server responds with either "Too low", "Too high" or "Correct".
- When a client wins the game, the server sends a message to all the other clients to inform them that the game is over and closes all TCP connections.
- A certain Duration after the server has started, it stops accepting new connections, sends to all its active connections a message saying that the game is ending and then closes all its connections.

Here is the implementation of the main task of the program:

```rust
use {
    async_broadcast::{Receiver, Sender},
    futures::future::Either,
    smol::{
        Timer,
        io::{AsyncBufReadExt, AsyncWriteExt},
        net::{TcpListener, TcpStream},
    },
    std::time::Duration,
};

#[derive(Clone, Copy)]
enum GameEndingEvent {
    TimeOut,
    SomeoneWon,
}

struct Server {
    /// Accepts incoming TCP connections
    listener: TcpListener,
    /// The server task will be informed of a game-ending event through this receiver
    event_receiver: Receiver<GameEndingEvent>,
    /// Used to inform other tasks of a game-ending event
    event_emitter: Sender<GameEndingEvent>,
}

impl Server {
    fn new(addr: &str, game_duration: Duration) -> Self {
        // `smol::net::TcpListener::bind` is an async function. At this point our program is
        // not doing anything else we can simply block the thread until it resolves.
        let listener = smol::block_on(TcpListener::bind(addr)).unwrap();
        let (event_emitter, event_receiver) = async_broadcast::broadcast(2);
        let timer_emitter = event_emitter.clone();
        smol::spawn(async move {
            Timer::after(game_duration).await;
            timer_emitter
                .broadcast(GameEndingEvent::TimeOut)
                .await
                .unwrap();
        })
        .detach();
        Self {
            listener,
            event_emitter,
            event_receiver,
        }
    }

    async fn run(&mut self) {
        loop {
            // Wait concurrently for either a game ending event or a new connection
            match futures::future::select(
                self.event_receiver.recv(),
                Box::pin(self.listener.accept()),
            )
            .await
            {
                Either::Left((_event, _)) => {
                    break;
                }
                Either::Right((stream, _)) => {
                    let (reader, writer) =
                        smol::io::split(stream.expect("Failed to accept connection").0);

                    // Create a connection handling task and spawn it
                    let connection = Connection {
                        reader: smol::io::BufReader::new(reader),
                        writer,
                        target: rand::random_range(1..=100),
                        event_emitter: self.event_emitter.clone(),
                        event_receiver: self.event_receiver.clone(),
                    };
                    smol::spawn(connection.handle()).detach();
                }
            }
        }
    }
}
```

The `Server` struct holds the listening socket, a broadcast channel end for game-ending events (`TimeOut` or `SomeoneWon`), and the corresponding receiver the main task will wait on.

Note `new` is an ordinary synchronous function, yet `TcpListener::bind` is asynchronous. We must therefore drive that future to completion with `smol::block_on`, which **blocks the current OS thread** until the listener is bound.
That is acceptable here: no other tasks exist yet, so nothing is starved while we wait.

Alternatively, we could have made `new` asynchronous by using the `async/await` syntax:

```rust
async fn new(addr: &str, game_duration: Duration) -> Self {
    let listener = TcpListener::bind(addr).await.unwrap();
    // ... same setup as before ...
}
```

Once the listener is bound, we _spawn a detached task_: it `await`s `Timer::after(game_duration)` and, when the duration elapses, _broadcasts_ `GameEndingEvent::TimeOut` on the shared channel.

Here we use:

- The `async` block syntax to declare the task.
- The [`smol::spawn`](https://docs.rs/smol/latest/smol/fn.spawn.html) method to poll the futures to which the block evaluates to completion.
- The [`smol::Task::detach`](https://docs.rs/smol/latest/smol/struct.Task.html#method.detach) method to add the task to the list of the tasks whose execution is scheduled by the runtime and resume the execution of the current task.
  Spawning a detached task can be seen as an equivalent of spawning a thread in a context where we delegate scheduling to the OS.

The **`run` method** is the connection-accepting loop. On each iteration it waits **concurrently** for either a message on `event_receiver` (a game-ending event, which breaks the loop) or an accepted TCP connection.
This is achieved by using the `futures::future::select` function, which drives concurrently two futures to completion and returns the result of the first one to complete.

When a client connects, we split the stream into read and write halves, build a `Connection` with the game state and channel handles, and **spawn** `connection.handle()` as its own task so accepting can continue while that client is served.

```rust
struct Connection {
    reader: smol::io::BufReader<smol::io::ReadHalf<TcpStream>>,
    writer: smol::io::WriteHalf<TcpStream>,
    /// The number that the client is trying to guess
    target: usize,
    /// The connection handler is informed of a game-ending event through this receiver
    event_receiver: Receiver<GameEndingEvent>,
    /// Used to inform other tasks of a game-ending event
    event_emitter: Sender<GameEndingEvent>,
}

impl Connection {
    async fn handle(mut self) {
        let mut guess_str = String::new();
        loop {
            guess_str.clear();
            match futures::future::select(
                self.reader.read_line(&mut guess_str),
                self.event_receiver.recv(),
            )
            .await
            {
                Either::Left((result, _)) => {
                    let n = result.unwrap();
                    if n == 0 {
                        continue;
                    }
                    if let Ok(guess) = guess_str.trim().parse::<usize>() {
                        if guess < self.target {
                            self.writer.write_all(b"Too low!\n").await.unwrap();
                        } else if guess > self.target {
                            self.writer.write_all(b"Too high!\n").await.unwrap();
                        } else {
                            self.writer.write_all(b"Correct!\n").await.unwrap();
                            self.event_emitter
                                .broadcast(GameEndingEvent::SomeoneWon)
                                .await
                                .unwrap();
                            break;
                        }
                    } else {
                        self.writer
                            .write_all(b"Invalid input. Please enter a number.\n")
                            .await
                            .unwrap();
                    }
                    continue;
                }
                Either::Right((event, _)) => {
                    match event.unwrap() {
                        GameEndingEvent::SomeoneWon => {
                            self.writer
                                .write_all(b"Game over! Someone else won.\n")
                                .await
                                .unwrap();
                        }
                        GameEndingEvent::TimeOut => {
                            self.writer
                                .write_all(b"Game over! Time's up.\n")
                                .await
                                .unwrap();
                        }
                    }
                    break;
                }
            }
        }
    }
}
```

The connection handling tasks are loops that at each iteration wait concurrently for either a full line from the client or a broadcast message from the server.
If the read completes first, we parse the line and react accordingly to the guess.
If the guess is correct we _broadcast_ `GameEndingEvent::SomeoneWon` to all other tasks so that the server will stop listening for new connections and the connection handling task will send a message to the client and exit the loop.

If the receiver completes first, some other part of the program has signaled the end of the game. We match on `SomeoneWon` vs `TimeOut`, send the appropriate “Game over!” line to the client, and exit the loop.

In order to run the program, we need to spawn the server task and block the main thread until it completes.

```rust
fn main() {
    let mut server = Server::new("127.0.0.1:8080", Duration::from_secs(300));
    smol::block_on(server.run());
}
```

When the program is running, we can connect to the server using several TCP clients that play the game in parallel—for example by opening **two terminals** and running `nc` in each:

{{< parallel-clients >}}

```bash
# Client 1
nc 127.0.0.1 8080
50
Too high!
25
Too high!
13
Too high!
8
Too low!
10
Too low!
12
Correct!
```

---PARALLEL---

```bash
# Client 2
nc 127.0.0.1 8080
50
Too high!
23
Too high!
Game over! Someone else won.
```

{{< /parallel-clients >}}

Note that this all works because the runtime is internally keeping track of what every task is waiting for and wakes them up when the dependencies are resolved.

In future posts we will dive into the details of the runtime and see how it works internally.
The whole implementation of the `smol`-based sever can be found [here](https://github.com/thenlevy/async_runtime/blob/master/src/with_smol/mod.rs)
