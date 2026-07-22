# chunk-tri-tree

A sparse binary triangle tree for adaptive terrain tiles.

Each `ChunkTriTree` covers one square tile, split along a diagonal into two
right-isosceles root triangles. A triangle is refined by bisecting its longest
edge (its "base"), so the tile is triangulated as an **RTIN** (Right-Triangulated
Irregular Network). Every node is addressed by a compact bit-path `NodeId` (or the
wider `NodeId64`, behind the `node64` feature), so the topology needs no child or
parent pointers — only materialised nodes are stored.

Vertex data lives in a single shared, reference-counted pool keyed by XZ position:
corners shared between neighbouring triangles resolve to one slot. `split` therefore
adds at most one vertex — the base midpoint, reused when a diamond neighbour has
already split — and merging frees that midpoint once both children release it.
Editing a corner is visible to every triangle that touches it with no propagation
step.

## Key operations

- `split` refines a leaf; `symmetric_split` also refines the base-edge (diamond)
  neighbour to keep the mesh free of T-junction cracks.
- `merge_flat` / `simplify_flat` collapse a split whose midpoint stays within a
  height tolerance of the parent's interpolated base edge.
- `id_at_xz`, `leaf_at_xz` and `height_at_xz` answer geometric point queries.

The tree is tile-local — cross-tile seams are the caller's responsibility.

## Features

| Feature | Default | Description |
|---------|:-------:|-------------|
| `std` | ✅ | Native `std` float math and `std::collections` maps. |
| `node64` | ✅ | Enables the 64-bit `NodeId64` id (depths up to 63). |
| `libm` |  | Forces `libm` math on *every* build for bit-reproducible cross-platform results. |
| `nostd-libm` |  | `no_std` build: `libm` math + `hashbrown` collections. |

### `no_std`

The crate is `no_std`-capable (heap allocation via `alloc` is still required):

```toml
[dependencies]
chunk-tri-tree = { version = "0.1", default-features = false, features = ["nostd-libm"] }
```

Add `node64` back to that feature list if you need the wider node id — it is a
default feature and so is dropped by `default-features = false`.

## Minimum supported Rust version

Rust **1.88** (edition 2024, let-chains, `i32::cast_signed`). Bumping the MSRV is
considered a breaking change.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
