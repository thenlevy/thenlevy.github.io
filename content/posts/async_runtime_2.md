---
date: "2026-04-19T19:00:00+02:00"
draft: false
title: "async_runtime Part II: Components of an asynchronous runtime"
description: "Building the core components of our asynchronous runtime (Executor and Reactor), and writing an asynchronous TCP server."
---

In [the previous post]({{< relref "/posts/async_runtime_1" >}}) we saw how asynchronous programming can be used to write concurrent programs in Rust.

In this post, we will start writing our own asynchronous runtime, and we will see an overview of its different components.
Our implementation is heavily inspired by this [series of posts](https://redixhumayun.github.io/async/2024/08/05/async-runtimes.html) and the actual code of the [`smol` runtime](https://github.com/smol-rs/).

# Components

The two main components of an asynchronous runtime are the `Executor` and the `Reactor`.

- The `Executor` is responsible for scheduling tasks and executing them. It contains the main execution loop of the runtime, which is responsible for polling the available tasks.
- The `Reactor` is responsible for registering resources that tasks are waiting for, and for waking the appropriate tasks when those resources are available.

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

            self.task_queue.borrow_mut().receive();

            if Reactor::waiting_on_events() {
                match Reactor::block_on_event_and_react() {
                    Ok(()) => {}
                    Err(e) => {
                        if e.kind() == std::io::ErrorKind::Interrupted {
                            break;
                        }
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
This means that if a task spawns a new task before completing, the runtime could be waiting for events that unlock other tasks even if the new one could already make progress.

To avoid this, the `Reactor` has a `notify` method that is called when spawning a task.

Calling this `notify` method ensures the `Reactor` is notified that at least one task can make progress.

Therefore the next time `block_on_event_and_react` is called, it will not hang if none of the events that it is waiting for are ready (but if some events are ready, it will wake up the tasks that are waiting on them).

{{< /callout >}}

We can see that the main loop algorithm can be summarized as follows:

1. Run all tasks that are ready to make progress.
2. Block until new tasks can make progress.

## The `Reactor`

Our `Reactor` is based on the [`epoll`](https://man7.org/linux/man-pages/man7/epoll.7.html) system call.
This system call is the key mechanism that will allow our tasks to be woken up when they can make progress.

{{< callout type="note" icon="🐧" title="Linux-specific implementation" >}}
Relying on `epoll`, which is a Linux-specific system call, means that the runtime is not portable to other OSes. Porting it to another OS means swapping the reactor for that platform’s multiplexing mechanism (for example `kqueue` on BSD/macOS or overlapped I/O on Windows).
{{< /callout >}}

What `epoll` offers is an interface to

- Register interest in events on file descriptors (typically readiness to read or write) without blocking with the [`epoll_ctl`](https://man7.org/linux/man-pages/man2/epoll_ctl.2.html) system call.
- Block until at least one of those events occurs with the [`epoll_wait`](https://man7.org/linux/man-pages/man2/epoll_wait.2.html) system call.

An `epoll`-based reactor then works by maintaining a single kernel-side set of registrations.
A call to `block_on_event_and_react` will then internally call `epoll_wait` which will sleep until at least one of the registered events occurs, and return all the descriptors that fired.
The reactor can then map those descriptors back to the tasks that were waiting on them and wake them up.

Here is a relevant extract of the implementation. Since we are using a low-level system call, the reader is invited to read the [full implementation](https://github.com/thenlevy/async_runtime/blob/master/src/a_la_mano/reactor.rs) for more details.

The reactor is a process-wide singleton (via a static `HANDLE`); the structs below are the core state: an `epoll` fd, a map from event keys to I/O sources, and a Unix socket pair used to wake the reactor when work is scheduled from outside the poll loop.

```rust
static HANDLE: AtomicPtr<Reactor> = AtomicPtr::new(core::ptr::null_mut());

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
```

Each `IoSource` wraps one non-blocking fd, tracks which tasks are waiting for read vs write readiness, and carries a stable `EventKey` that `epoll` echoes back in `epoll_event.u64` so wakeups can be routed without relying on the raw fd number alone.

Registering a new fd adds it to the epoll set with `EPOLL_CTL_ADD` and stores an `IoSource` in `sources`. User-visible interest updates go through `register_interest`, which issues `EPOLL_CTL_MOD` with a bitmask derived from our `Event` type. The implementation always sets `EPOLLONESHOT`, so each edge of readiness consumes the registration and the reactor must re-arm after handling an event (see `block_on_event_and_react` below).

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EventKey(u64);

#[derive(Debug)]
pub struct Event {
    key: EventKey,
    pub readable: bool,
    pub writable: bool,
}

impl Reactor {
    pub fn add_source(fd: Rc<OwnedFd>) -> std::io::Result<Rc<RefCell<IoSource>>> {
        let this = Self::get();
        this.add_source_inner(fd)
    }

    fn add_source_inner(&mut self, fd: Rc<OwnedFd>) -> std::io::Result<Rc<RefCell<IoSource>>> {
        let key = EventKey::new();
        let epoll_event: EpollEvent = Event::none(key).into();
        let mut libc_epoll_event: libc::epoll_event = epoll_event.into();

        let ret = unsafe {
            epoll_ctl(
                self.epoll_fd.as_raw_fd(),
                libc::EPOLL_CTL_ADD,
                fd.as_ref().as_raw_fd(),
                (&mut libc_epoll_event) as *mut libc::epoll_event,
            )
        };

        if ret == -1 {
            return Err(std::io::Error::last_os_error());
        }
        let new_source = Rc::new(RefCell::new(IoSource {
            fd: fd,
            key,
            readers: Vec::new(),
            writers: Vec::new(),
        }));
        self.sources.insert(key, new_source.clone());
        Ok(new_source)
    }

    pub fn register_interest(fd: BorrowedFd<'_>, interest: Event) -> std::io::Result<()> {
        let this = Self::get();
        this.register_interest_inner(fd, interest)
    }

    fn register_interest_inner(&self, fd: BorrowedFd<'_>, interest: Event) -> std::io::Result<()> {
        let epoll_event: EpollEvent = interest.into();
        let mut libc_epoll_event: libc::epoll_event = epoll_event.into();

        let ret = unsafe {
            epoll_ctl(
                self.epoll_fd.as_raw_fd(),
                libc::EPOLL_CTL_MOD,
                fd.as_raw_fd(),
                (&mut libc_epoll_event) as *mut libc::epoll_event,
            )
        };

        if ret == -1 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }
}
```

`EventKey::new()` hands out unique keys from a global counter. `Event` is the logical “what we care about” on a source; it is converted to `libc::epoll_event` (including `EPOLLONESHOT` and `EPOLLIN` / `EPOLLOUT`) elsewhere in the file.

The heart of blocking is `epoll_wait`: the reactor allocates a buffer of `epoll_event` structs, waits indefinitely (`timeout = -1`), maps errors so `EINTR` yields zero events, then converts each kernel event back into our `Event` type via the embedded `u64` key.

```rust
impl Reactor {
    const MAX_EVENT: u32 = 1024;

    fn wait_for_events(&self) -> std::io::Result<Vec<Event>> {
        let mut events = vec![libc::epoll_event { events: 0, u64: 0 }; Self::MAX_EVENT as usize];
        let nb_events = {
            let epoll_result = unsafe {
                libc::epoll_wait(
                    self.epoll_fd.as_raw_fd(),
                    events.as_mut_ptr(),
                    Self::MAX_EVENT as i32,
                    -1,
                )
            };

            if epoll_result < 0 {
                Err(std::io::Error::last_os_error())
            } else {
                Ok(epoll_result)
            }
        }
        .or_else(|e| {
            if e.kind() == std::io::ErrorKind::Interrupted {
                // `epoll_wait` can return EINTR when a signal arrives before any fd is ready.
                Ok(0)
            } else {
                Err(e)
            }
        })?;

        Ok(events
            .iter()
            .take(nb_events as usize)
            .map(|event| Event::from(EpollEvent::from(*event)))
            .collect())
    }
}
```

This is the function the executor calls when it has nothing ready to poll: it sleeps until the kernel reports readiness (or a signal interrupts the syscall).

After `epoll_wait` returns (via `wait_for_events`), `block_on_event_and_react` propagates real I/O errors and returns immediately if the batch is empty—for example when `epoll_wait` was interrupted (`EINTR`) and `wait_for_events` maps that to zero events. Otherwise it looks up each `EventKey`, drains the wakers for the readiness directions that fired, recomputes what the source still needs (`waiting_for`), and re-registers interest with `register_interest` when the source still has pending waiters. Finally it wakes the tasks and drains the notify pipe so the self-pipe wakeup mechanism stays balanced.

```rust
impl Reactor {
    pub fn block_on_event_and_react() -> std::io::Result<()> {
        let this = Self::get();

        // Interests to be re-registered after the one-shot epoll_wait call.
        let mut interests = Vec::new();

        let events = this.wait_for_events()?;
        if events.is_empty() {
            return Ok(());
        }

        // The wakers to be woken up.
        let mut wakers = Vec::with_capacity(events.len());
        for event in events {
            if let Some(mut source) = this.sources.get(&event.key).map(|rc| rc.borrow_mut()) {
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
        for waker in wakers {
            waker.wake();
        }

        this.clear_spawn_notifications()?;
        Ok(())
    }
}
```

Together, `wait_for_events` and `block_on_event_and_react` close the loop: the executor blocks on the kernel until something is ready, then translates readiness into `Waker::wake` calls so polled futures can run again.

# Implementing a TCP server

## An asynchronous `TcpListener`

Now that we have the core components of our runtime, we can start implementing a TCP server.

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
```

`AsyncTcpListener::bind` creates a normal blocking `TcpListener`, switches it to _non-blocking_ mode.
It then hands ownership of the socket to the reactor via `OwnedFd` and `Reactor::add_source`.
From that point on, the reactor tracks the underlying fd in its epoll set, allowing tasks to register **wakers** on that source when they need to wait for “readable” (for a listening socket, that means a connection may be available to accept).

{{< callout type="note" icon="👉" >}}
Note that calling `accept()` does not perform I/O by itself. It only builds a fresh `TcpConnectionAccept` future in the `Start` state.

This is a standard pattern for async Rust: the method does not perform anything by itself, but only returns a `Future`, which needs to be polled to perform the actual operation.
{{< /callout >}}

### Implementing `Future` for `TcpConnectionAccept`

As is often the case, the implementation of the `Future` trait for `TcpConnectionAccept` relies on an internal state machine, and the `poll` method is a simple match statement that delegates to the appropriate method based on the current state.

```rust
#[derive(Debug, Clone, Copy)]
enum TcpConnectionAcceptState {
    Start,
    FirstAttemptBlocked,
    WokenWhenReady,
    Finished,
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

Initially our future is in the `Start` state and tries to call `accept()` once. If the kernel has a pending connection, it returns `Poll::Ready` immediately.
Otherwise, it calls `poll_first_attempt_blocked` to register its waker as a reader on the `IoSource` associated with the socket and returns `Pending` after setting the state to `WokenWhenReady`.

When our future is in the `WokenWhenReady` state, it calls `poll_assume_ready` when polled. Here it assumes that a call to `accept()` cannot block again and treats `WouldBlock` as a logic bug (`panic!`).

```rust
impl TcpConnectionAccept {
    fn poll_start(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<std::io::Result<(TcpStream, SocketAddr)>> {
        let source = self.source.borrow();
        // SAFETY: The fd of self.source is a valid TCP listener fd.
        let tcp_listener = unsafe { TcpListener::from_raw_fd(source.get_raw_fd()) };

        std::mem::drop(source);

        let ret = tcp_listener.accept();
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

```

{{< callout type="note" icon="🪤" title="Why wrap `TcpListener` around a raw fd each time">}}
The future does not keep a `TcpListener` field. It reconstructs one with `TcpListener::from_raw_fd` whenever it needs to call `accept()`, then **detaches** it with `into_raw_fd()` before dropping. The socket is owned by `OwnedFd` inside the reactor and the temporary `TcpListener` is just a typed view on it. Dropping it without `into_raw_fd()` would close the fd and break every other reference to the same listener.

{{< /callout >}}

## An asynchronous `TcpStream`

### Reading lines from a `TcpStream` asynchronously

Accepting a connection yields a `TcpStream`. The next step is to read from it without blocking the executor: same pattern as the listener—non-blocking socket, `OwnedFd`, reactor registration.

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
        Ok(Self { _inner: fd, source })
    }

    pub fn get_lines(&self) -> TcpStreamLines<'_> {
        TcpStreamLines::new(self)
    }
}
```

`AsyncTcpStream` mirrors `AsyncTcpListener::bind`: the stream is switched to non-blocking mode, wrapped in `Rc<OwnedFd>`, and registered with `Reactor::add_source`.
Like `accept()`, `get_lines()` performs no I/O by itself and only constructs a `TcpStreamLines` that will read when polled.

`TcpStreamLines` uses a similar pattern to [`BufReader`](https://doc.rust-lang.org/std/io/struct.BufReader.html) over [`Read`](https://doc.rust-lang.org/std/io/trait.Read.html): it **batches** reads into a fixed buffer (`buf`), tracks consumed bytes (`pos`, `cap`), and assembles lines across refills (`next_line`).
A `BorrowedFd` ties the helper’s lifetime to the underlying `AsyncTcpStream` so the raw fd view stays valid.

```rust
const BUF_SIZE: usize = 4096;

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
}
```

The actual implementation lives in `poll_line`, a loop that alternates refilling the buffer and scanning it.
Whenever the in-memory window `[pos, cap)` is empty, we need another kernel `read`.
Here, we use the same pattern as we did to asynchronously `accept` a TCP connection: we build a temporary `TcpStream` with `from_raw_fd`, call `read`, then **detach** with `into_raw_fd` so we never close the shared fd.

```rust
// Inside `poll_line` — refill when the buffer window is empty
if self.pos >= self.cap {
    // SAFETY: `inner` borrows the open `TcpStream` fd for the lifetime of `TcpStreamLines`.
    let mut stream = unsafe { TcpStream::from_raw_fd(self.inner.as_raw_fd()) };
    self.cap = match stream.read(self.buf.as_mut_slice()) {
        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
            if let Err(e) = self.source.borrow_mut().add_reader(cx.waker().clone()) {
                let _ = stream.into_raw_fd();
                return Poll::Ready(Some(Err(e)));
            }

            // Drop the stream without closing the associated file
            let _ = stream.into_raw_fd();
            return Poll::Pending;
        }
        ret => {
            self.pos = 0;
            let _ = stream.into_raw_fd();
            ret
        }
    }?;
}
```

If `read` returns `WouldBlock`, we register the task’s waker as a **reader** on the `IoSource` and return `Poll::Pending`; the reactor will wake us when data arrives. A successful read updates `cap` (and resets `pos`); the surrounding `loop` in `poll_line` then either parses a line from `[pos, cap)` or runs this refill again.

```rust
// Still inside the same `loop` in `poll_line`, after a non-empty buffer
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
            "invalid UTF-8",
        ))));
    }
} else {
    self.next_line
        .extend_from_slice(&self.buf[self.pos..self.cap]);
    self.pos = self.cap;
}
```

If `cap == 0` we have reached the end of the file and return `None`.
Otherwise, if we find a newline character (`\n`), we append all the bytes up to (but not including) the newline to `next_line` and return the line.
Finally, if we do not find a newline, we append the remaining bytes to `next_line` and continue the loop.

{{< callout type="note" icon="📌" title="Async `next` and the `Stream` trait" >}}
In the `futures` crate, [`Stream`](https://docs.rs/futures/latest/futures/stream/trait.Stream.html) is the asynchronous counterpart of [`Iterator`](https://doc.rust-lang.org/std/iter/trait.Iterator.html): the stream itself is polled for the _next_ item via [`poll_next`](https://docs.rs/futures/latest/futures/stream/trait.Stream.html#tymethod.poll_next), which returns `Poll<Option<T>>`. The [`StreamExt::next`](https://docs.rs/futures/latest/futures/stream/trait.StreamExt.html#method.next) helper wraps that in a [`Future`](https://doc.rust-lang.org/std/future/trait.Future.html), so consuming a stream in async code looks like repeated `await`s—exactly the “async function” feel of `lines.next()`.

While `TcpStreamLines` does not implement `futures::Stream` directly (because we want to keep dependencies minimal), it exposes a similar interface: each call to `next()` returns a fresh `TcpLinesNext` future whose `Output` is an `Option<…>`.

The standard library’s [`AsyncIterator`](https://doc.rust-lang.org/std/async_iter/trait.AsyncIterator.html) (still unstable) standardizes the `Stream` interface.
{{< /callout >}}

Finally, `next` packages `poll_line` behind a `Future` so call sites can `await` one line at a time.

```rust
impl<'s> TcpStreamLines<'s> {
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

Each `TcpLinesNext` is cheap: it only holds a mutable borrow of the line buffer and forwards `poll` to `poll_line`, where the state machine and buffer live.

### Writing to a `TcpStream` asynchronously

To write to a `TcpStream` asynchronously, we follow the same pattern as reading lines: we build a `Future` that will perform the write when it is polled. When polled, this future will attempt to write its entire buffer to the stream, and register its waker as a writer on its `IoSource` when the write would block.

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
        // SAFETY: `inner` borrows the open `TcpStream` fd for the lifetime of `TcpStreamWriteAll`.
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

Now that we have the core components of our asynchronous runtime, we can implement a simple TCP server. It runs on a single thread, multiplexes multiple connections, and depends on the `libc` crate only for the `epoll` syscalls.

```rust
async fn run() {
    let tcp_listener =
        tcp::AsyncTcpListener::bind("127.0.0.1:8080").expect("Failed to bind TCP listener");

    loop {
        let (tcp_stream, _addr) = tcp_listener
            .accept()
            .await
            .expect("Failed to accept TCP connection");

        if let Err(e) = Executor::spawn(handle_connection(
            tcp::AsyncTcpStream::from_tcp_stream(tcp_stream).unwrap(),
        )) {
            println!("Failed to spawn task: {e}");
        }
    }
}


async fn handle_connection(stream: tcp::AsyncTcpStream) {
    let mut lines = stream.get_lines();

    while let Some(line) = lines.next().await {
        let Ok(line) = line.inspect_err(|e| {
            eprintln!("Error while reading line: {e:?}");
        }) else {
            continue;
        };

        let reply = format!("{}!!!\n", line.to_uppercase());
        if let Err(e) = stream.write_all(reply.as_bytes()).await {
            eprintln!("Error while writing line back to client: {e:?}");
        }
    }
}

pub fn start() {
    Executor::block_on(run());
}
```
