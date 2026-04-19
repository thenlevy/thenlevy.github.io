---
date: "2026-04-19T19:00:00+02:00"
draft: false
title: "async_runtime Part II: Components of an asynchronous runtime"
description: "Building the core components of our asynchronous runtime, and writing an asynchronous TCP server."
---

In [the previous post]({{< relref "/posts/async_runtime_1" >}}) we saw how asynchronous programming can be used to write concurrent programs in Rust.

In this post, we will start writing our own asynchronous runtime, and we will see an overview of it's different components.
Our implementation is heavily inpired by this [series of posts](https://redixhumayun.github.io/async/2024/08/05/async-runtimes.html) and the actual code of the [`smol` runtime](https://github.com/smol-rs/).

# Components

The two main components of an asynchronous runtime are the `Executor` and the `Reactor`.

- The `Executor` is responsible for scheduling tasks and executing them. It it contains the main execution loop of the runtime, that is responsible for polling the available tasks.
- The `Reactor` is responsible for registering ressources that tasks are waiting for, and to wake up the appropriate tasks when the ressources are available.

## The `Executor`

Here is the main loop of our `Executor`. Note that we are purposefully writing a single-threaded runtime, so we allow our `Executor` to be `!Send` and `!Sync`.

```rust
pub struct Executor {
    task_queue: Rc<RefCell<TaskQueue>>,
    next_task_id: Cell<usize>,
}

impl Executor {
    fn run(&self) {
        loop {
            self.task_queue.borrow_mut().receive();

            // Run all tasks that are ready to make progress.
            println!("Running {} tasks", self.task_queue.borrow().len());
            loop {
                let Some(task) = self.task_queue.borrow_mut().pop() else {
                    break;
                };

                let waker = MyWaker::new(Rc::clone(&task), self.task_queue.borrow().sender());
                let mut context = Context::from_waker(&waker);
                match task.future.borrow_mut().as_mut().poll(&mut context) {
                    std::task::Poll::Ready(_output) => {}
                    std::task::Poll::Pending => {}
                };
            }

            eprintln!("Recieving tasks");
            self.task_queue.borrow_mut().receive();
            eprintln!(
                "After running tasks, {} tasks remain",
                self.task_queue.borrow().len()
            );

            if Reactor::waiting_on_events() {
                match Reactor::block_on_event_and_react() {
                    Ok(()) => {}
                    Err(e) => {
                        if e.kind() == std::io::ErrorKind::Interrupted {
                            break;
                        }
                        eprintln!("Error while waiting for IO events :{}", e);
                    }
                }
            } else if self.task_queue.borrow().is_empty() {
                // No task is waiting and no task can be run. We can exit the execution loop
                break;
            }
        }
    }
}
```

{{< callout type="note" icon="🪤" title="Note on blocking on the reactor" >}}
Notice that the execution loop blocks on the `Reactor::block_on_event_and_react` call.
This means that if a tasks spawns a new task before completing, the runtime could be waiting for events that unlocks other tasks even if the new one could already make progress.

To avoid this, the `Reactor` has a `notify` method that is called when spawning a task.

Calling this `notify` method ensures that the `Reactor` tells it that at least one task can make progress.

Therefore the next time `block_on_event_and_react` is called, it will not hang if none of the events that it is waiting for are ready (but if some events are ready, it will wake up the tasks that are waiting on them).

{{< /callout >}}

We can see that the main loop alogorithm can be summarized as follows:

1. Run all tasks that are ready to make progress.
2. Block until new tasks can make progress.

## The `Reactor`

Our `Reactor` is based on the [`epoll`](https://man7.org/linux/man-pages/man7/epoll.7.html) system call.
This system call is the key mechanism that will allow our task to be woken up when they can make progress.

What `epoll` offers is an interface to

- Register interest in events on file descriptors (typically readiness to read or write) without blocking with the [`epoll_ctl`](https://man7.org/linux/man-pages/man2/epoll_ctl.2.html) system call.
- Block until at least one of those events occurs with the [`epoll_wait`](https://man7.org/linux/man-pages/man2/epoll_wait.2.html) system call.

An `epoll`-based reactor then works by maintaining a single kernel-side set of registrations.
A call to `block_on_event_and_react` will then internally call `epoll_wait` which will sleep until at least one of the registered events occurs, and return all the descriptors that fired.
The reactor can then map those descriptors back to the tasks that were waiting on them and wake them up.

Here is a relevant extract of the implementation. Since we are using a low-level system call, the reader is invited to read the [full implementation](https://github.com/thenlevy/async_runtime/blob/master/src/a_la_mano/reactor.rs) for more details.

```rust
pub struct Reactor {
    epoll_fd: OwnedFd,
    sources: HashMap<EventKey, Rc<RefCell<IoSource>>>,
    notify_stream: UnixStream,
    notify_event_key: Option<EventKey>,
}

pub struct IoSource {
    fd: Rc<OwnedFd>,
    key: EventKey,
    // Wakers for tasks that are waiting for this source to be readable or writable.
    readers: Vec<Waker>,
    writers: Vec<Waker>,
}

impl Reactor {

    pub fn block_on_event_and_react() -> std::io::Result<()> {
        let this = Self::get();
        let mut events = vec![libc::epoll_event { events: 0, u64: 0 }; Self::MAX_EVENT as usize];

        // Interests to be re-registered after the one-shot epoll_wait call.
        let mut interests = Vec::new();

        let res = {
            let nb_events = unsafe {
                libc::epoll_wait(
                    this.epoll_fd.as_raw_fd(),
                    events.as_mut_ptr(),
                    Self::MAX_EVENT as i32,
                    -1,
                )
            };

            if nb_events == -1 {
                Err(std::io::Error::last_os_error())
            } else {
                Ok(nb_events)
            }
        };
        match res {
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => Ok(()),
            Err(e) => Err(e),
            Ok(nb_events) => {
                assert!(nb_events >= 0);
                // The wakers to be woken up.
                let mut wakers = Vec::with_capacity(nb_events as usize);
                let events = &events[0..nb_events as usize];
                for event in events {
                    let epoll_event = EpollEvent::from(*event);
                    let event = Event::from(epoll_event);
                    eprintln!("got event {event:?}");

                    if let Some(mut source) = this.sources.get(&event.key).map(|rc| rc.borrow_mut())
                    {
                        // If the event is readable, add all the wakes that are waiting for it to be
                        // readable. If the event is writable, add all the
                        // wakes that are waiting for it to be writable.
                        if event.readable {
                            source.drain_readers_into(&mut wakers);
                        }
                        if event.writable {
                            source.drain_writers_into(&mut wakers);
                        }
                        let event = source.waiting_for();

                        // If the source is still waiting for further events, we must re-register
                        // the interest.
                        if event.readable || event.writable {
                            interests.push((source.fd.clone(), event));
                        }
                    }
                }
                for (fd, interest) in interests {
                    Self::register_interest(fd.as_fd(), interest).unwrap();
                }
                eprintln!("Waking {} tasks", wakers.len());
                for waker in wakers {
                    waker.wake();
                }

                // Clear the spawn notifications see the callout above 🪤 Note on blocking on the reactor
                {
                    // ....
                }
                Ok(())
            }
        }
    }
}
```

# Implementing a TCP server

## An asynchronous `TcpListener`

Now that we have the core components of our runtime we can start implementing a TCP server.

For that, we need to implement an asynchronous `TcpListener`. This will be the opportunity to see how to implement the `Future` trait.

```rust
pub struct AsyncTcpListener {
    _inner: Rc<OwnedFd>,
    source: Rc<RefCell<IoSource>>,
}

impl AsyncTcpListener {
    pub fn bind(addr: &str) -> std::io::Result<Self> {
        let listener = TcpListener::bind(addr)?;
        listener.set_nonblocking(true)?;
        let fd = Rc::new(OwnedFd::from(listener));
        let source = Reactor::add_source(fd.clone())?;
        Ok(Self { _inner: fd, source })
    }

    pub fn accept(&self) -> TcpConnectionAccept {
        TcpConnectionAccept {
            state: TcpConnectionAcceptState::Start,
            source: self.source.clone(),
        }
    }
}

pub struct TcpConnectionAccept {
    source: Rc<RefCell<IoSource>>,
    state: TcpConnectionAcceptState,
}

// This allows to convert `&mut self` into a `Pin<&mut Self>`.
impl Unpin for TcpConnectionAccept {}

#[derive(Debug, Clone, Copy)]
enum TcpConnectionAcceptState {
    Start,
    FirstAttemptBlocked,
    WokenWhenReady,
    Finished,
}

impl TcpConnectionAccept {
    fn poll_start(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<std::io::Result<(TcpStream, SocketAddr)>> {
        println!("Polling TcpConnectionAccept in Start state");

        let source = self.source.borrow();
        // SAFETY: The fd of self.source is a valid TCP listener fd.
        let tcp_listener = unsafe { TcpListener::from_raw_fd(source.get_raw_fd()) };

        std::mem::drop(source);

        let ret = tcp_listener.accept();
        println!("First accept attempt returned: {:?}", ret);
        match ret {
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                self.state = TcpConnectionAcceptState::FirstAttemptBlocked;
                // Drop the TcpListener without closing the fd.
                let _ = tcp_listener.into_raw_fd();
                self.poll_first_attempt_blocked(cx)
            }
            res => {
                // Drop the TcpListener without closing the fd.
                let _ = tcp_listener.into_raw_fd();
                Poll::Ready(res)
            }
        }
    }

    fn poll_first_attempt_blocked(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<std::io::Result<(TcpStream, SocketAddr)>> {
        if let Err(e) = self.source.borrow_mut().add_reader(cx.waker().clone()) {
            return Poll::Ready(Err(e));
        };
        self.state = TcpConnectionAcceptState::WokenWhenReady;
        Poll::Pending
    }

    fn poll_assume_ready(
        &mut self,
        _cx: &mut Context<'_>,
    ) -> Poll<std::io::Result<(TcpStream, SocketAddr)>> {
        // SAFETY: The fd of self.source is a valid TCP listener fd.
        let tcp_listener = unsafe { TcpListener::from_raw_fd(self.source.borrow().get_raw_fd()) };

        match tcp_listener.accept() {
            Ok(ret) => {
                // Drop the TcpListener without closing the fd.
                let _ = tcp_listener.into_raw_fd();
                self.state = TcpConnectionAcceptState::Finished;
                Poll::Ready(Ok(ret))
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                panic!("TcpListener was not actually ready");
            }
            Err(e) => {
                // Drop the TcpListener without closing the fd.
                let _ = tcp_listener.into_raw_fd();
                Poll::Ready(Err(e))
            }
        }
    }
}

impl Future for TcpConnectionAccept {
    type Output = std::io::Result<(TcpStream, SocketAddr)>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut this = self.as_mut();
        match this.state {
            TcpConnectionAcceptState::Start => this.poll_start(cx),
            TcpConnectionAcceptState::FirstAttemptBlocked => this.poll_first_attempt_blocked(cx),
            TcpConnectionAcceptState::WokenWhenReady => this.poll_assume_ready(cx),
            TcpConnectionAcceptState::Finished => {
                panic!("poll called after TcpConnectionAccept is finished")
            }
        }
    }
}
```

`AsyncTcpListener::bind` creates a normal blocking `TcpListener`, switches it to _non-blocking_ mode.
It then hands ownership of the socket to the reactor via `OwnedFd` and `Reactor::add_source`.
From that point on, the reactor tracks the underlying fd in its epoll set, allowing tasks to register **wakers** on that source when they need to wait for “readable” (for a listening socket, that means a connection may be available to accept).

{{< callout type="note" icon="👉" >}}
Note that calling `accept()` does not perform I/O by itself. It only builds a fresh `TcpConnectionAccept` future in the `Start` state.

This is a standard pattern for async rust: The methods does not perform anything by itself, but only returns a Future, which needs to be polled to perform the actual operation.
{{< /callout >}}

### Implementing `Future` for `TcpConnectionAccept`

As is often the case, the implementation of the `Future` trait for `TcpConnectionAccept` relies on an internal state machine.

In thise case, the `poll` method is a simple match statement that delegates to the appropriate method based on the current state.

Initially our future is in the `Start` state and tries to call `accept()` once. If the kernel has a pending connection, it returns `Poll::Ready` immediately.
Otherwise, it calls `poll_first_attempt_blocked` to register it's waker as a reader on the `IoSource` associated to the socket and returns `Pending` after setting the state to `WokenWhenReady`.

When our future is in the `WokenWhenReady` state, it calls `poll_assume_ready` when polled. Here it assumes that a call to `accept()` cannot block again and treats `WouldBlock` as a logic bug (`panic!`).

{{< callout type="note" icon="🪤" title="Why wrap `TcpListener` around a raw fd each time">}}
The future does not keep a `TcpListener` field. It reconstructs one with `TcpListener::from_raw_fd` whenever it needs to call `accept()`, then **detaches** it with `into_raw_fd()` before dropping. The socket is owned by `OwnedFd` inside the reactor and the temporary `TcpListener` is just a typed view on it. Dropping it without `into_raw_fd()` would close the fd and break every other reference to the same listener.

{{< /callout >}}

## An asynchronous `TcpStream`

### Reading lines from a `TcpStream` asynchronously

What we got by accepting a connection is a `TcpStream`. We need to be able to read lines from it asynchronously.

```rust

pub struct AsyncTcpStream {
    _inner: Rc<OwnedFd>,
    source: Rc<RefCell<IoSource>>,
}

impl AsyncTcpStream {
    pub fn from_tcp_stream(stream: TcpStream) -> std::io::Result<Self> {
        stream.set_nonblocking(true)?;

        let fd = Rc::new(OwnedFd::from(stream));
        let source = Reactor::add_source(fd.clone())?;
        dbg!("AsyncTcpStream fd: {}", source.borrow().get_raw_fd());
        Ok(Self { _inner: fd, source })
    }

    pub fn get_lines(&self) -> TcpStreamLines<'_> {
        TcpStreamLines::new(self)
    }
}

pub struct TcpStreamLines<'s> {
    inner: BorrowedFd<'s>,
    source: Rc<RefCell<IoSource>>,
    buf: Box<[u8; BUF_SIZE]>,
    pos: usize,
    cap: usize,
    next_line: Vec<u8>,
}

impl<'s> TcpStreamLines<'s> {
    fn new(stream: &'s AsyncTcpStream) -> Self {
        Self {
            inner: stream._inner.as_ref().as_fd(),
            source: stream.source.clone(),
            buf: Box::new([0; BUF_SIZE]),
            pos: 0,
            cap: 0,
            next_line: Vec::new(),
        }
    }

    fn poll_line(&mut self, cx: &mut Context<'_>) -> Poll<Option<std::io::Result<String>>> {
        loop {
            eprintln!("Buffer pos: {}, cap: {}", self.pos, self.cap);
            // If we have consumed all the bytes of the buffer, fill it
            if self.pos >= self.cap {
                // SAFETY `self._inner` is open because we own a valid reference to it and is a
                // valid fd for a TcpStream.
                let mut stream = unsafe { TcpStream::from_raw_fd(self.inner.as_raw_fd()) };
                eprintln!("Poll reading");
                self.cap = match stream.read(self.buf.as_mut_slice()) {
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        eprintln!("Would block");
                        if let Err(e) = self.source.borrow_mut().add_reader(cx.waker().clone()) {
                            let _ = stream.into_raw_fd();
                            return Poll::Ready(Some(Err(e)));
                        }

                        // Drop the stream without closing the associated file
                        let _ = stream.into_raw_fd();
                        return Poll::Pending;
                    }
                    ret => {
                        eprintln!("Ready");
                        self.pos = 0;
                        let _ = stream.into_raw_fd();
                        ret
                    }
                }?;
            }
            if self.cap == 0 {
                return Poll::Ready(None);
            }

            if let Some(i) = self.buf[self.pos..self.cap]
                .iter()
                .position(|b| *b == b'\n')
            {
                // Do not take the \n
                self.next_line
                    .extend_from_slice(&self.buf[self.pos..(self.pos + i)]);
                self.pos += i + 1;
                if let Ok(mut ret) = String::from_utf8(std::mem::take(&mut self.next_line)) {
                    if ret.ends_with('\r') {
                        ret.pop();
                    }
                    return Poll::Ready(Some(Ok(ret)));
                } else {
                    return Poll::Ready(Some(Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "Not utf8",
                    ))));
                }
            } else {
                self.next_line
                    .extend_from_slice(&self.buf[self.pos..self.cap]);
                self.pos = self.cap;
            }
        }
    }

    pub fn next<'a>(&'a mut self) -> TcpLinesNext<'a, 's> {
        TcpLinesNext { lines: self }
    }
}

pub struct TcpLinesNext<'a, 's> {
    lines: &'a mut TcpStreamLines<'s>,
}

impl<'a, 's> Future for TcpLinesNext<'a, 's> {
    type Output = Option<std::io::Result<String>>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = Pin::get_mut(self);

        this.lines.poll_line(cx)
    }
}
```

`AsyncTcpStream` mirrors the listener setup: the underlying `TcpStream` is made non-blocking, wrapped in `OwnedFd`, and registered with the reactor through `Reactor::add_source`.
The stream keeps both the `Rc<OwnedFd>` (so the socket stays open) and the `Rc<RefCell<IoSource>>` (so tasks can subscribe for “readable” wakeups).

In the same way that `AsyncTcpListener::accept()` does not read anything by itself, the `AsyncTcpStream::get_lines()` method does not read anything by itself: It only builds a `TcpStreamLines` value that can produce lines asynchronously when its `next` method is called.

{{< callout type="note" icon="📌" title="Async `next` and the `Stream` trait" >}}
In the `futures` crate, [`Stream`](https://docs.rs/futures/latest/futures/stream/trait.Stream.html) is the asynchronous counterpart of [`Iterator`](https://doc.rust-lang.org/std/iter/trait.Iterator.html): the stream itself is polled for the _next_ item via [`poll_next`](https://docs.rs/futures/latest/futures/stream/trait.Stream.html#tymethod.poll_next), which returns `Poll<Option<T>>`. The [`StreamExt::next`](https://docs.rs/futures/latest/futures/stream/trait.StreamExt.html#method.next) helper wraps that in a [`Future`](https://doc.rust-lang.org/std/future/trait.Future.html), so consuming a stream in async code looks like repeated `await`s—exactly the “async function” feel of `lines.next()`.

While `TcpStreamLines` does not implement `futures::Stream` directly (because we want to keep depencies minimal), it exposes a similar interface: each call to `next()` returns a fresh `TcpLinesNext` future whose `Output` is an `Option<…>`.

The standard library’s [`AsyncIterator`](https://doc.rust-lang.org/std/async_iter/trait.AsyncIterator.html) (still unstable) standardizes the `Stream` interface.
{{< /callout >}}

`TcpStreamLines` plays the same role over our non-blocking socket that [`BufReader`](https://doc.rust-lang.org/std/io/struct.BufReader.html) plays over any [`Read`](https://doc.rust-lang.org/std/io/trait.Read.html):
it **batches** kernel reads into an internal buffer area and looks for newlines in the buffer before asking for more bytes.
If a newline could not be found in the buffer, we **refill** it with one more asychronous `read`.
This read is performed in a similar way to how we asynchronously accepted connection on our `TcpListener`:
If `read` returns `WouldBlock`, we register the current waker as a reader on the `IoSource` and return `Poll::Pending`.

### Writing to a `TcpStream` asynchronously

To write to a `TcpStream` asynchronously, we follow the same pattern as reading lines: we build a `Future` that will perform the write when it's polled. When polled, this futures will attempt to write its entire buffer to the stream, and register it's waker as a writer of it's `IoSource` when the write would block.

```rust
impl AsyncTcpStream {
    pub fn write_all(&self, buf: &[u8]) -> impl Future<Output = std::io::Result<()>> {
        TcpStreamWriteAll::new(self, buf)
    }
}

struct TcpStreamWriteAll<'s, 'b> {
    inner: BorrowedFd<'s>,
    buf: &'b [u8],
    source: Rc<RefCell<IoSource>>,
}

impl<'s, 'b> TcpStreamWriteAll<'s, 'b> {
    fn new(stream: &'s AsyncTcpStream, buf: &'b [u8]) -> Self {
        Self {
            inner: stream._inner.as_ref().as_fd(),
            buf,
            source: stream.source.clone(),
        }
    }
}

impl<'s, 'b> Future for TcpStreamWriteAll<'s, 'b> {
    type Output = std::io::Result<()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = Pin::get_mut(self);
        // SAFETY `self._inner` is open because we own it and is a valid fd for a TcpStream.
        let mut stream = unsafe { TcpStream::from_raw_fd(this.inner.as_raw_fd()) };

        while !this.buf.is_empty() {
            match stream.write(this.buf) {
                Ok(0) => {
                    let _ = stream.into_raw_fd();
                    return Poll::Ready(Err(std::io::Error::new(
                        std::io::ErrorKind::WriteZero,
                        "failed to write whole buffer",
                    )));
                }
                Ok(n) => {
                    this.buf = &this.buf[n..];
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    let _ = stream.into_raw_fd();
                    if let Err(e) = this.source.borrow_mut().add_writer(cx.waker().clone()) {
                        return Poll::Ready(Err(e));
                    }
                    return Poll::Pending;
                }
                Err(e) => {
                    let _ = stream.into_raw_fd();
                    return Poll::Ready(Err(e));
                }
            }
        }
        let _ = stream.into_raw_fd();
        Poll::Ready(Ok(()))
    }
}
```

# Putting it all together: A simple TCP server

Now that we have the core components of our asynchronous runtime, we can start implementing a simple TCP server. This server runs asynchronously on a single thread and can handle multiple connections, and has a single dependency on the `libc` crate to handle the `epoll` syscalls.

```rust
async fn run() {
    let tcp_listener =
        tcp::AsyncTcpListener::bind("127.0.0.1:8080").expect("Failed to bind TCP listener");

    loop {
        eprintln!("Waiting for incoming TCP connections...");
        let (tcp_stream, _addr) = tcp_listener
            .accept()
            .await
            .expect("Failed to accept TCP connection");

        println!("Accepted connection from {:?}", tcp_stream.peer_addr());
        if let Err(e) = Executor::spawn(handle_connection(Rc::new(
            tcp::AsyncTcpStream::from_tcp_stream(tcp_stream).unwrap(),
        ))) {
            println!("Failed to spawn task: {e}");
        }
    }
}

async fn handle_connection(stream: Rc<tcp::AsyncTcpStream>) {
    let mut lines = stream.get_lines();

    while let Some(line) = lines.next().await {
        let Ok(mut line) = line.inspect_err(|e| {
            println!("Error while reading line: {:?}", e);
        }) else {
            continue;
        };

        let spawn_handle = Rc::clone(&stream);
        Box::pin(async move {
            line = format!("{}!!!\n", line.to_uppercase());
            spawn_handle
                .write_all(line.as_bytes())
                .await
                .inspect_err(|e| {
                    println!("Error while writing line back to client: {e:?}");
                })
                .ok();
        })
        .await;
    }
}

pub fn start() {
    Executor::block_on(run());
}
```
