---
date: "2026-10-29T21:00:00+01:00"
draft: false
title: "`bonzai` Part I: How Postgres lays out a table on disk"
description: "Finding the file behind a Postgres table, reading its first 8 KB page byte by byte, then fetching and parsing pages in Rust."
---

I have started [`bonzai`](https://github.com/thenlevy/bonzai): a small B-tree engine that mimics Postgres's on-disk format. My goal with this project,
is to understand the internals of some of Postgres's features (MVCC, crash recovery, etc..) by implementing them myself.
As I progress, I want to write posts about the concepts that I will learn along the way, and show how they translate into code.

I this article, we look at how Postgres lays out its data on disk, and we implement utilities to read and parse Postgres pages from the file system.

# Looking at a table on disk

## Setting up a cluster

Beffore digging into code, let's start by inspecting files actually created by Postgres.
We start by creating a new _cluster_ (i.e. a collections of databases managed by a Postgres instance), and starting an instance to manage it.
Then we create the `bonzai` database and connect to it:

```sh
# Create a new cluster
# --data-checksums makes Postgres store a checksum in every page, and is off by default before Postgres 18.
initdb -D data --data-checksums -U postgres -A trust
# Start the postgres instance
pg_ctl -D data -l log start
# Create the database
createdb -U postgres bonzai
# Connect to the database with the psql client
psql -U postgres -d bonzai
```

Using the `psql` client, we create a table with a few rows, and we use the `CHECKPOINT` command to force the changes to be written to disk.
This is needed because, in order to optimize disk IO, Postgres will only modify it's in-memory version of the data after the `INSERT`.
In "regular use" the data would be written to the disk later, `CHECKPOINT` forces that to happen now.

```sql
CREATE TABLE t (id int8 GENERATED ALWAYS AS IDENTITY, payload text);
INSERT INTO t (payload) VALUES ('Hello, World!'), ('Bonjour Bonsai'), ('An other row');
CHECKPOINT;
```

Note that, despite keeping the modified data in-memory, Postgress is still able to gurantee data recovery after a crash.
We will see how when we explore the functioning of the WAL.

## Finding the file holding our table's data

Everything a cluster stores lives under its `data` directory, and each database gets a subdirectory of `base/` named after the database's OID (Object IDentifier):

```text
bonzai=# SELECT oid, datname FROM pg_database;
  oid  |  datname
-------+-----------
     5 | postgres
 16384 | bonzai
     1 | template1
     4 | template0
(4 rows)
```

Each _relation_ (table, index, sequence, ...) of a database is associated with up to four _forks_:
The _main fork_, that contains the actual data stored in the relation, and optionally the _free space map fork_ (`fsm`), the _visibility map fork_ (`vm`), and the _init fork_.

We'll explore the `fsm` and `vm` forks when we discuss how `VACUUM` works.

Forks are named after the relation's `relfilenode`, which can be found in the `pg_class` catalog:

```text
bonzai=# SELECT oid, relname, relfilenode FROM pg_class WHERE relname = 't';
  oid  | relname | relfilenode
-------+---------+-------------
 16386 | t       |       16386
(1 row)
```

This tells us that the main fork of our table is called `16386`.

Big forks of over 1GB are split into _segments_ each stored as a file of up to 1 GB.
The first segment is stored in a file whose name is exactly the name of the fork, and the names of subsequent segments are suffixed with `.{idx}` where `idx` starts at 1 for the 2nd segment: `16386`, `16386.1`, `16386.2`, ...
See [Database File Layout](https://www.postgresql.org/docs/current/storage-file-layout.html) in the Postgres documentation for the details.

{{< callout type="note" icon="🕳️" title="OID or relfilenode?" >}}
For now the table's `relfilenode` is equal to its OID, but the two diverge as soon as the table is rewritten (by a `TRUNCATE` or a `VACUUM FULL` for example): the new version of the table goes into a new file and the old file is deleted.
{{< /callout >}}

For now our table is small, so we expect its data to live in `data/base/16384/16386`.

We could also have reached this conclusion using the `pg_relation_filepath` function, which immediately gives us the result.

```text
bonzai=# SELECT pg_relation_filepath('t');
 pg_relation_filepath
----------------------
 base/16384/16386
(1 row)
```

Forks segments, are sequences of blocks of `BLCKSZ` bytes, where `BLCKSZ` is a constant chosen when Postgres is compiled. In practice `BLCKSZ` is almost always 8192 bytes, and this is the value we'll use in `bonzai`.

Block `n` of a fork starts at position `n * BLCKSZ` bytes from the start of the fork (counting across its segments).

{{< callout type="note" icon="📄" title="Block or page?" >}}
Postgres tends to use the word _block_ when talking about a position in a file and _page_ when referring to its content, but the two words are often used interchangeably.
{{< /callout >}}

## The first page

Our table is small, so it has only one block. It's main fork is therefore stored in a single file that is 8192 bytes long:

```text
$ wc -c data/base/16384/16386
8192 data/base/16384/16386
```

Let's look at it:

```text
$ hexdump -C data/base/16384/16386
00000000  00 00 00 00 a8 42 b6 01  1b 34 00 00 24 00 70 1f  |.....B...4..$.p.|
00000010  00 20 04 20 00 00 00 00  d0 9f 5c 00 a0 9f 5e 00  |. . ......\...^.|
00000020  70 9f 5a 00 00 00 00 00  00 00 00 00 00 00 00 00  |p.Z.............|
00000030  00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00  |................|
*
00001f70  e4 02 00 00 00 00 00 00  00 00 00 00 00 00 00 00  |................|
00001f80  03 00 02 00 02 08 18 00  03 00 00 00 00 00 00 00  |................|
00001f90  1b 41 6e 20 6f 74 68 65  72 20 72 6f 77 00 00 00  |.An other row...|
00001fa0  e4 02 00 00 00 00 00 00  00 00 00 00 00 00 00 00  |................|
00001fb0  02 00 02 00 02 08 18 00  02 00 00 00 00 00 00 00  |................|
00001fc0  1f 42 6f 6e 6a 6f 75 72  20 42 6f 6e 73 61 69 00  |.Bonjour Bonsai.|
00001fd0  e4 02 00 00 00 00 00 00  00 00 00 00 00 00 00 00  |................|
00001fe0  01 00 02 00 02 08 18 00  01 00 00 00 00 00 00 00  |................|
00001ff0  1d 48 65 6c 6c 6f 2c 20  57 6f 72 6c 64 21 00 00  |.Hello, World!..|
00002000
```

Before we dive into the details of how to interpret those bytes, we can already nottice that the file has the following structure:

- There are non zero bytes at the very beggining of the file
- Then a long empty region
- Then a non empty region, that notably contains the three strings that we inserted into the table, in reverse order of insertion.

# Anatomy of a page ([PR #1](https://github.com/thenlevy/bonzai/pull/1))

All pages share the same layout, described in [bufpage.h](https://github.com/postgres/postgres/blob/master/src/include/storage/bufpage.h) and summarized in the documentation of `bonzai`'s [`page` module](/bonzai/part-1/doc/bonsai/storage/page/index.html):

```text
+--------+---------------+----------------+--------------+---------+
| header | line pointers |→  free space  ←| tuple data   | special |
+--------+---------------+----------------+--------------+---------+
0        24          pd_lower         pd_upper     pd_special    8192
```

- A 24-byte **header**.
- An array of **line pointers** of 4 bytes each, one per tuple, which grows forward.
- The **tuples** (versions of rows), which are added from the end of the page and grow backward.
- The **free space** in between. The page is full when the two meet.
- At the very end, a **special space** reserved for the access method. This space is for example used by B-Tree indexes to embbed their topology.
  Heap pages (pages that store the data) do not use it.

In our hexdump above, the "non zero region at the beginning" corresponds to the header and the line pointers, and the "non empty region at the end" corresponds to the tuples.

| Bytes            | Content                    |
| ---------------- | -------------------------- |
| `0x0000..0x0018` | the header                 |
| `0x0018..0x0024` | three line pointers        |
| `0x0024..0x1f70` | free space                 |
| `0x1f70..0x2000` | three tuples, newest first |

## The header

In `bonzai`, a [`Page`](/bonzai/part-1/doc/bonsai/storage/page/struct.Page.html) is nothing but its bytes, and its header is parsed into a [`Header`](/bonzai/part-1/doc/bonsai/storage/page/struct.Header.html) whose fields mirror Postgres's `PageHeaderData` (most doc comments are omitted in the excerpts of this post):

```rust
/// An 8 KB block, exactly as stored on disk.
pub struct Page {
    pub bytes: [u8; BLCKSZ],
}

pub struct Header {
    pub lsn: Option<XLogRecPtr>,
    pub checksum: u16,
    pub flags: PageFlags,
    pub lower: LocationIndex,
    pub upper: LocationIndex,
    pub special: LocationIndex,
    pub pagesize_version: u16,
    pub prune_xid: Option<TransactionId>,
}
```

Here is how these fields map to the first 24 bytes of our page:

| Offset | Size | Field                 | Bytes                     | Value            |
| -----: | ---: | --------------------- | ------------------------- | ---------------- |
|      0 |    8 | `pd_lsn`              | `00 00 00 00 a8 42 b6 01` | `0/1B642A8`      |
|      8 |    2 | `pd_checksum`         | `1b 34`                   | 13339            |
|     10 |    2 | `pd_flags`            | `00 00`                   | no flag set      |
|     12 |    2 | `pd_lower`            | `24 00`                   | 36               |
|     14 |    2 | `pd_upper`            | `70 1f`                   | 8048             |
|     16 |    2 | `pd_special`          | `00 20`                   | 8192             |
|     18 |    2 | `pd_pagesize_version` | `04 20`                   | 8196 = 8192 \| 4 |
|     20 |    4 | `pd_prune_xid`        | `00 00 00 00`             | none             |

In a few words:

- `pd_lsn` is used to locate the page in the WAL.
- `pd_checksum` is the page's checksum.
- `pd_flags` holds hints about properties of the page.
- `pd_lower`, `pd_upper` and `pd_special` are the boundaries of the diagram above.
- `pd_pagesize_version` packs the page size and the version of the layout (4, since Postgres 8.3) into a single 16-bit integer.
- `pd_prune_xid` tells Postgres when pruning the page might be worth it. Nothing was deleted yet, so there is nothing to prune.

The [rustdocs](/bonzai/part-1/doc/bonsai/storage/page/struct.Header.html) of the project goes into more details.

### Parsing the header

Fields are stored in the byte order of the machine, `bonzai` assumes little-endian (see `read_u16` bellow).
How the LSN is parsed is a bit convoluted for historical reasons:
The LSN used to be handled as two 32-bit integers, a "log file" number and an offset within that file, so that the WAL could grow past 4 GB using only 32-bit integers.
It has been a single 64-bit integer since Postgres 9.3, but pages still store the two halves so that the on-disk format did not change.

```rust
impl Header {
    /// Parses a header, with the checks Postgres runs before letting a page
    /// into the buffer pool. They don't prove the header right, only plausible.
    pub fn try_from_raw(raw: &[u8; HEADER_SIZE]) -> Result<Self, InvalidHeader> {
        // Postgres stores the LSN as two 32-bit halves, high half first.
        let lsn = (u64::from(read_u32(raw, 0)) << 32) | u64::from(read_u32(raw, 4));
        let header = Header {
            lsn: XLogRecPtr::from_raw(lsn),
            checksum: read_u16(raw, 8),
            flags: PageFlags::try_from(read_u16(raw, 10))?,
            lower: LocationIndex(read_u16(raw, 12)),
            upper: LocationIndex(read_u16(raw, 14)),
            special: LocationIndex(read_u16(raw, 16)),
            pagesize_version: read_u16(raw, 18),
            prune_xid: TransactionId::from_raw(read_u32(raw, 20)),
        };
        let Header { lower, upper, special, .. } = header;
        if !(lower <= upper
            && upper <= special
            && usize::from(special) <= BLCKSZ
            && usize::from(special) % MAXIMUM_ALIGNOF == 0)
        {
            return Err(InvalidHeader::Offsets { lower, upper, special });
        }
        Ok(header)
    }
}

fn read_u16(raw: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([raw[offset], raw[offset + 1]])
}
// fn read_u32 follows the same pattern
```

Postgres does not trust what it reads from disk.
Before letting a page into the buffer pool, [`PageIsVerified`](https://github.com/postgres/postgres/blob/cab16d214bc5b70dd3514c5631ffa1f0acb3fed3/src/backend/storage/page/bufpage.c#L63-L174) checks that its header is plausible: no unknown bit is set in `pd_flags`, `pd_lower <= pd_upper <= pd_special <= BLCKSZ`, and `pd_special` is aligned.
`try_from_raw` runs the same checks, the one on flags being done by `PageFlags::try_from`, which refuses bits Postgres doesn't define.

Postgres uses the value 0 to mean "invalid" for both LSNs and transaction IDs.
In `bonzai`, these types wrap non-zero integers, so the invalid value becomes `None` and the `Option` is free:

```rust
pub struct XLogRecPtr(NonZeroU64);

impl XLogRecPtr {
    /// The LSN stored as `raw`, or `None` if `raw` is 0.
    pub fn from_raw(raw: u64) -> Option<Self> {
        NonZeroU64::new(raw).map(XLogRecPtr)
    }
}

const _: () = assert!(size_of::<Option<XLogRecPtr>>() == size_of::<u64>());
```

Reading our page with these types gives:

```text
Header {
    lsn: Some(XLogRecPtr(0x1B642A8)),
    checksum: 13339,
    flags: PageFlags { has_free_lines: false, page_full: false, all_visible: false },
    lower: LocationIndex(36),
    upper: LocationIndex(8048),
    special: LocationIndex(8192),
    pagesize_version: 8196,
    prune_xid: None,
}
```

## Checksums

With data checksums enabled, every page carries a 16-bit checksum of its content, computed when the page is written to disk and verified when it is read back.
It is meant to detect corruption caused by the storage stack (disk, controller, file system), not to resist tampering.

For Bonzai, we use an AI translation of the C code in [checksum_impl.h](https://github.com/postgres/postgres/blob/master/src/include/storage/checksum_impl.h).

Two details are worth noting:

- The block number is part of the checksum, so a perfectly valid page written at the wrong place in the file is detected. Our page validates as block 0, but as block 1 the computed checksum would be 13340 instead of 13339.
- A checksum is never 0, so that 0 can mean "this page has no checksum".

```rust
/// The checksum `page` should carry when stored at `block`. The page's own
/// `pd_checksum` field is read as zero.
pub fn checksum_page(page: &Page, block: BlockNumber) -> u16 {
    // Mixing in the block number catches a valid page written to the wrong place.
    let checksum = checksum_block(page) ^ block.0;
    // Fold to 16 bits, plus one so that a checksum is never zero.
    (checksum % 65535 + 1) as u16
}
```

{{< callout type="note" icon="🔍" title="Where is the checksum?" >}}
If you inspect the page with `pageinspect` right after the `CHECKPOINT`, you will find no checksum:

```text
bonzai=# SELECT lsn, checksum, lower, upper FROM page_header(get_raw_page('t', 0));
    lsn    | checksum | lower | upper
-----------+----------+-------+-------
 0/1B642A8 |        0 |    36 |  8048
(1 row)
```

`get_raw_page` reads the page through the buffer pool, and the copy of the page that lives there has never had a checksum.
When Postgres writes a page, it computes the checksum on a private copy of the page (other processes may be setting hint bits concurrently, which would invalidate it), writes that copy, and leaves the buffer untouched.

After restarting the server, the page is read back from disk, and the checksum shows up.
{{< /callout >}}

## Validated pages

The checks described so far are bundled in [`Page::try_validated_for_block`](/bonzai/part-1/doc/bonsai/storage/page/struct.Page.html#method.try_validated_for_block), the counterpart of Postgres's [`PageIsVerified`](https://github.com/postgres/postgres/blob/cab16d214bc5b70dd3514c5631ffa1f0acb3fed3/src/backend/storage/page/bufpage.c#L63-L174).
It returns a `ValidatedPage`, a page whose checksum and header have been checked for a given block.
When Postgres extends a relation, it first writes pages of zeros to the file and initializes them in memory later, so a crash can leave all-zero pages behind. Such pages are therefore valid, even though they have no checksum:

```rust
pub fn try_validated_for_block<'a>(
    &'a self,
    block: BlockNumber,
) -> Result<ValidatedPage<'a>, PageError> {
    if self.is_uninitialized() {
        return if self.iter().all(|&byte| byte == 0) {
            Ok(ValidatedPage { page: Cow::Borrowed(self), block })
        } else {
            Err(PageError::InvalidHeader { block })
        };
    }
    // Read on its own, so that a checksum mismatch is reported even when
    // `header()` rejects the header.
    let stored = read_u16(self, PD_CHECKSUM.start);
    let computed = checksum_page(self, block);
    if stored != computed {
        return Err(PageError::ChecksumMismatch { block, stored, computed });
    }
    if self.header().is_err() {
        return Err(PageError::InvalidHeader { block });
    }
    Ok(ValidatedPage { page: Cow::Borrowed(self), block })
}
```

On the write side, [`Page::try_set_checksum_for_block`](/bonzai/part-1/doc/bonsai/storage/page/struct.Page.html#method.try_set_checksum_for_block) refuses pages with an invalid header, stamps the checksum and returns a `ValidatedPage` too.

# Testing framework ([PR #1](https://github.com/thenlevy/bonzai/pull/1))

Let's test our implementation against Postgres: we should be able to read pages written by Postrgres, parse their headers and check that their checksums match.

To that end, we create a small testing framework, made of three pieces:

- **a script**, `make-db.sh`, which builds a fresh cluster and runs the fixtures against it;
- **fixtures**, SQL scripts that set up some state in the cluster and record what Postgres says about it;
- **Rust tests**, which load the pages Postgres wrote along with what it said about them, and compare both to what `bonzai` reads.

## Building a cluster

The scripts [`tests/fixtures/make-db.sh`](https://github.com/thenlevy/bonzai/blob/e41e161d3f9617ac9da8c68c2bd6c8b20833edc9/tests/fixtures/make-db.sh)
rebuilds a cluster from scratch in the git-ignored `postgres/` directory.
It then runs every `tests/fixtures/*.sql` script, storing postgres's output in `postgres/<name>.facts`, and stops the server cleanly so that every page is on disk.

## Recording what Postgres sees

The first fixture, [`postgres-heap-page.sql`](https://github.com/thenlevy/bonzai/blob/e41e161d3f9617ac9da8c68c2bd6c8b20833edc9/tests/fixtures/postgres-heap-page.sql), creates the same table as our walkthrough, and dumps the header of its first page with `pageinspect`:

```sql
CREATE TABLE t (id int8 GENERATED ALWAYS AS IDENTITY, payload text);
INSERT INTO t (payload) VALUES ('Hello, World!'), ('Bonjour Bonsai'), ('An other row');

-- Set the hint bits on the rows before the page is written.
SELECT count(*) FROM t \gset

-- Commit the fork state to disk, which sets the checksum on the page.
CHECKPOINT;

-- Dump the page's header in human readable format.
CREATE EXTENSION pageinspect;
SELECT pg_relation_filepath('t') AS path,
    lsn - '0/0' AS lsn,
    checksum::int & 65535 AS checksum,
    flags,
    lower,
    upper,
    special,
    pagesize | version AS pagesize_version,
    prune_xid
FROM page_header(pg_read_binary_file(pg_relation_filepath('t'), 0, 8192));
```

A few things in there deserve an explanation:

- The page is read with `pg_read_binary_file`, straight from the file, because the copy of the page in the buffer pool has no checksum.
- `lsn - '0/0'` turns the LSN into a plain number, and `checksum::int & 65535` turns the checksum, which `page_header` returns as a signed 16-bit integer, back into an unsigned one.
- Unlike our walkthrough, the fixture reads the table before the `CHECKPOINT`.

{{< callout type="note" icon="✍️" title="Reads that write" >}}
When a row is inserted, its header records the ID of the transaction that inserted it, but not whether that transaction committed.
The first query that reads the row after the commit looks the transaction's status up in the commit log (`pg_xact`), and caches the answer in the row's header as a _hint bit_, so that later readers don't have to.

Setting a hint bit modifies the page.
Without the `SELECT count(*)`, the first query to read `t` after the dump (for instance while looking around the cluster) would set the hint bits, the page would be written again with a new LSN and checksum, and the file would no longer match the saved facts.
In our walkthrough, which had no such `SELECT`, the two bytes `02 08` in each tuple header become `02 09` once a query has read the row.
{{< /callout >}}

`psql` is run in a mode that prints one `name=value` line per column, so the resulting `postgres-heap-page.facts` looks like this:

```text
path=base/5/16385
lsn=24527312
checksum=5983
flags=0
lower=36
upper=8048
special=8192
pagesize_version=8196
prune_xid=0
```

## Comparing with `bonzai`

On the Rust side, [`tests/postgres.rs`](https://github.com/thenlevy/bonzai/blob/e41e161d3f9617ac9da8c68c2bd6c8b20833edc9/tests/postgres.rs) reads the facts, loads block 0 of the file they point to, and builds the header `pageinspect` saw:

```rust
static POSTGRES: LazyLock<Fixture> = LazyLock::new(|| {
    let facts = Facts::read("postgres-heap-page");
    let path: String = facts.get("path");
    let file = read(&format!("data/{path}"));
    let block = file
        .get(..BLCKSZ)
        .unwrap_or_else(|| panic!("{path} is shorter than a block"));
    Fixture {
        page: Page {
            bytes: block.try_into().unwrap(),
        },
        header: Header {
            lsn: XLogRecPtr::from_raw(facts.get("lsn")),
            checksum: facts.get("checksum"),
            // ...
        },
    }
});
```

The tests then check that `bonzai` reads the same header as `pageinspect`, computes the same checksum as Postgres, and writes back exactly the bytes Postgres wrote when it seals the page again:

```rust
#[test]
fn header_reads_like_pageinspect() {
    assert_eq!(POSTGRES.page.header(), Ok(POSTGRES.header));
}

#[test]
fn checksum_matches_postgres() {
    assert_ne!(POSTGRES.header.checksum, 0, "checksums are off");
    assert_eq!(
        checksum_page(&POSTGRES.page, BlockNumber(0)),
        POSTGRES.header.checksum
    );
    assert!(
        POSTGRES
            .page
            .try_validated_for_block(BlockNumber(0))
            .is_ok()
    );
}

#[test]
fn sealing_writes_back_what_postgres_wrote() {
    let mut page = POSTGRES.page.clone();
    let mut header = page.header().unwrap();
    header.checksum = 0;
    page.set_header(header);
    let sealed = page.try_set_checksum_for_block(BlockNumber(0)).unwrap();
    assert_eq!(sealed.block(), BlockNumber(0));
    assert_eq!(sealed.page(), &POSTGRES.page);
}
```

A last test checks that the page fails validation as block 1.

# Reading blocks from a fork ([PR #2](https://github.com/thenlevy/bonzai/pull/2))

In the tests above, the page is read from the file system directly.
In Postgres, Fetching the page from the file system, is the job of the _storage manager_ ([`smgr.c`](https://github.com/postgres/postgres/blob/master/src/backend/storage/smgr/smgr.c) and [`md.c`](https://github.com/postgres/postgres/blob/master/src/backend/storage/smgr/md.c)).

In `bonzai`, this is done by the [`StorageManager`](/bonzai/part-1/doc/bonsai/storage/manager/struct.StorageManager.html).

A block is addressed by its relation, its fork and its block number:

```rust
pub struct RelFileNumber(pub u32);

pub enum ForkKind {
    Main = 0,
    FreeSpaceMap = 1,
    VisibilityMap = 2,
    Init = 3,
}

pub struct ForkIdentifier {
    pub rel: RelFileNumber,
    pub kind: ForkKind,
}

pub struct BlockIdentifier {
    pub fork: ForkIdentifier,
    pub block_number: BlockNumber,
}
```

A `StorageManager` is opened on the directory of a database (`base/<database OID>`), and turns these addresses into a file and an offset, following the naming rules we saw above:

```rust
/// Path to the file holding the segment `segno` of `fork`.
fn path(&self, fork: ForkIdentifier, segno: u32) -> PathBuf {
    let mut name = fork.rel.0.to_string();
    match fork.kind {
        ForkKind::Main => {}
        ForkKind::FreeSpaceMap => name.push_str("_fsm"),
        ForkKind::VisibilityMap => name.push_str("_vm"),
        ForkKind::Init => name.push_str("_init"),
    }
    if segno > 0 {
        write!(&mut name, ".{segno}").unwrap();
    }
    self.dir.join(name)
}

fn position(&self, block: BlockNumber) -> Position {
    let segno = block.0 / self.n_block_per_segment.0;
    let offset = u64::from(block.0 % self.n_block_per_segment.0) * BLCKSZ as u64;
    Position { segno, offset_bytes: offset }
}
```

The subtle part is knowing how many blocks a fork has.
Like Postgres's [`mdnblocks`](https://github.com/postgres/postgres/blob/cab16d214bc5b70dd3514c5631ffa1f0acb3fed3/src/backend/storage/smgr/md.c#L1224-L1285), `bonzai` opens the segments in order and adds up their sizes, stopping after the first segment that isn't full, since only the last segment of a fork can be partly full.
A partly written block at the end of a file, which a crash in the middle of an extension can leave behind, doesn't count, and a segment larger than the segment size is an error.
The segments of a fork are opened and counted once, then cached, and reading a block checks that it exists before reading it at its offset:

```rust
/// Fetch the block from the disk and store its bits into `page`.
pub fn read(&self, block: BlockIdentifier, page: &mut Page) -> Result<(), BlockError> {
    let fork = self.get_fork(block.fork).map_err(|err| BlockError {
        block,
        inner: BlockErrorInner::ForkError(err),
    })?;
    if !fork.nb_blocks.contains(block.block_number) {
        return Err(BlockError {
            block,
            inner: BlockErrorInner::BlockDoesNotExist {
                fork_size: fork.nb_blocks,
            },
        });
    }
    let Position { segno, offset_bytes } = self.position(block.block_number);
    fork.segments[segno as usize]
        .read_exact_at(page, offset_bytes)
        .map_err(|err| BlockError {
            block,
            inner: BlockErrorInner::Io(err),
        })
}
```

Putting the two PRs together, fetching and validating the page of our table takes a few lines:

```rust
let smgr = StorageManager::open("data/base/16384").unwrap();
let block = BlockIdentifier {
    fork: ForkIdentifier {
        rel: RelFileNumber(16386),
        kind: ForkKind::Main,
    },
    block_number: BlockNumber(0),
};
let mut page = Page { bytes: [0; BLCKSZ] };
smgr.read(block, &mut page).unwrap();
let validated = page.try_validated_for_block(block.block_number).unwrap();
```

`validated.page().header()` gives back the header we parsed above, and reading block 1 fails with `BlockDoesNotExist { fork_size: BlockCount(1) }`: our table has a single block.

PR #2 also comes with a test that does the same for every block Postgres wrote, [`tests/storage_manager.rs`](https://github.com/thenlevy/bonzai/blob/549d6ccd65389cd4151e51ab39f523fe7673e0ac/tests/storage_manager.rs).
It needs no fixture of its own: it opens a `StorageManager` on the `postgres` database of the cluster built by `make-db.sh`, and reads every block of every fork in it.
These are the system catalogs and their B-tree indexes, their free space and visibility maps, and the relations created by the fixtures, 300 forks and 929 blocks in all.
Each block must have the same bytes as in its file and pass `try_validated_for_block`, and reading one block past the end of the fork must fail:

```rust
for entry in fs::read_dir(FIXTURE_DB).unwrap() {
    let path = entry.unwrap().path();
    let Some(fork) = fork_of(&path) else {
        continue;
    };
    let file = fs::read(&path).unwrap();
    let nb_blocks = (file.len() / BLCKSZ) as u32;
    for (n, expected) in (0..).zip(file.chunks_exact(BLCKSZ)) {
        let page = read(&smgr, fork, n).unwrap();
        assert!(page.bytes == expected, "{path:?}: block {n} differs");
        if let Err(err) = page.try_validated_for_block(BlockNumber(n)) {
            panic!("{path:?}: {err}");
        }
    }
    assert!(
        is_past_the_end(&read(&smgr, fork, nb_blocks), nb_blocks),
        "{path:?}"
    );
}
```

Here `fork_of` turns a file name such as `16385_fsm` into a `ForkIdentifier`, and skips other files such as `pg_filenode.map`.

# What's next

We can now read any block of a fork and tell whether Postgres would accept it, but we know nothing yet about what lies between `pd_lower` and `pd_upper`.
Future posts will look at line pointers and tuples, then at writing and extending forks, the buffer pool, the WAL, and of course B-trees.
