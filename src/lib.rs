//! A sparse binary triangle tree for adaptive terrain tiles.
//!
//! Each [`ChunkTriTree`] covers one square tile, split along a diagonal into
//! two right-isosceles root triangles. A triangle is refined by bisecting its
//! longest edge (its "base"), so the tile is triangulated as an RTIN
//! (Right-Triangulated Irregular Network). Every node is addressed by a compact
//! bit-path [`NodeId`] (or the wider `NodeId64`, behind the `node64` feature),
//! so the topology needs no child or parent pointers — only materialised
//! nodes are stored.
//!
//! Vertex data lives in a single shared, reference-counted pool keyed by XZ
//! position: corners shared between neighbouring triangles resolve to one slot.
//! [`ChunkTriTree::split`] therefore adds at most one vertex — the base
//! midpoint, reused when a diamond neighbour has already split — and merging
//! frees that midpoint once both children release it. Editing a corner is
//! visible to every triangle that touches it with no propagation step.
//!
//! Key operations:
//! - [`ChunkTriTree::split`] refines a leaf; [`ChunkTriTree::symmetric_split`]
//!   also refines the base-edge (diamond) neighbour to keep the mesh free of
//!   T-junction cracks.
//! - [`ChunkTriTree::merge_flat`] / [`ChunkTriTree::simplify_flat`] collapse a
//!   split whose midpoint stays within a height tolerance of the parent's
//!   interpolated base edge.
//! - [`ChunkTriTree::id_at_xz`], [`ChunkTriTree::leaf_at_xz`] and
//!   [`ChunkTriTree::height_at_xz`] answer geometric point queries.
//!
//! The tree is tile-local — cross-tile seams are the caller's responsibility
//! (see [`ChunkTriTree::symmetric_split`]). The crate depends only on [`glam`]
//! for math; the `libm` / `nostd-libm` features select the math backend (e.g.
//! `libm` for bit-reproducible cross-platform results).
//!
//! # `no_std`
//!
//! The crate is `no_std`-capable: build with `--no-default-features --features
//! nostd-libm` to drop the `std` dependency. Heap allocation is still required
//! (the tree owns `Vec`/`HashMap` storage via `alloc`), and `nostd-libm`
//! supplies both `libm` float math and `hashbrown` collections. Note that
//! `--no-default-features` also turns off `node64`; add it back if you need the
//! wider [`NodeId`] variant.

#![cfg_attr(not(feature = "std"), no_std)]

// `no_std` builds must be told where float math and hash maps come from.
#[cfg(all(not(feature = "std"), not(feature = "nostd-libm")))]
compile_error!(
    "`no_std` builds (default feature `std` disabled) require the `nostd-libm` \
     feature, which provides `libm` float math and `hashbrown` collections"
);

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use glam::{Vec2, Vec3};

#[cfg(not(feature = "std"))]
use hashbrown::{HashMap, HashSet};
#[cfg(feature = "std")]
use std::collections::{HashMap, HashSet};

/// Bit-path identifier into the triangle subdivision tree.
///
/// The root square is implicit. A leading `1` sentinel bit precedes the
/// path bits, read MSB-first after the sentinel:
///
/// - First path bit: which triangle of the square (`0` or `1`).
/// - Each subsequent bit: which child after a split (`0` = left, `1` = right).
///
/// ```text
///   0b1_0  (2) — triangle 0
///   0b1_1  (3) — triangle 1
///   0b1_00 (4) — triangle 0, left child
///   0b1_01 (5) — triangle 0, right child
///   0b1_10 (6) — triangle 1, left child
///   0b1_11 (7) — triangle 1, right child
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

/// 64-bit variant of [`NodeId`], allowing depths up to 63 path bits.
///
/// Encoding is identical (sentinel `1` bit followed by MSB-first path bits);
/// only the storage width differs. [`ChunkTriTree`] keys on the 32-bit
/// [`NodeId`]; use `NodeId64` for trees or callers that need deeper paths.
///
/// Gated behind the `node64` feature.
#[cfg(feature = "node64")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId64(pub u64);

/// Maximum depth a `NodeId` can represent (sentinel bit at bit 31).
const MAX_DEPTH: u32 = 31;

/// Maximum depth a `NodeId64` can represent (sentinel bit at bit 63).
#[cfg(feature = "node64")]
const MAX_DEPTH_64: u32 = 63;

macro_rules! impl_node_id {
    ($Name:ident, $Repr:ty, $MaxDepth:ident) => {
        impl $Name {
            /// The two top-level triangles that form the root square.
            pub const TRI_0: Self = Self(0b10);
            pub const TRI_1: Self = Self(0b11);

            /// Depth of this node (number of path bits). Depth 1 = top-level triangle.
            #[must_use]
            pub const fn depth(self) -> u32 {
                <$Repr>::BITS - 1 - self.0.leading_zeros()
            }

            /// Parent node. Returns `None` for the two top-level triangles.
            #[must_use]
            pub const fn parent(self) -> Option<Self> {
                if self.0 <= 3 {
                    None
                } else {
                    Some(Self(self.0 >> 1))
                }
            }

            /// Left child (appends a `0` bit).
            #[must_use]
            pub const fn left(self) -> Self {
                debug_assert!(self.depth() < $MaxDepth);
                Self(self.0 << 1)
            }

            /// Right child (appends a `1` bit).
            #[must_use]
            pub const fn right(self) -> Self {
                debug_assert!(self.depth() < $MaxDepth);
                Self(self.0 << 1 | 1)
            }

            /// Children pair `[left, right]`.
            #[must_use]
            pub const fn children(self) -> [Self; 2] {
                [self.left(), self.right()]
            }

            /// Whether this is the left child of its parent (last path bit is `0`).
            #[must_use]
            pub const fn is_left(self) -> bool {
                self.0 & 1 == 0
            }

            /// Which top-level triangle this node belongs to (`0` or `1`).
            #[must_use]
            pub const fn root_triangle(self) -> u32 {
                ((self.0 >> (self.depth() - 1)) & 1) as u32
            }

            /// Iterate the path bits from root to this node (MSB-first, excluding sentinel).
            pub fn path_bits(self) -> impl Iterator<Item = u32> {
                let d = self.depth();
                (0..d).rev().map(move |i| ((self.0 >> i) & 1) as u32)
            }

            /// All node ids that would exist if this node were reached by recursive
            /// subdivision from the root triangle.
            ///
            /// Each subdivision creates both children. The result includes the root
            /// triangle, both children at every interior level, and the target's
            /// sibling.
            #[must_use]
            pub fn subdivision_ids(self) -> Vec<Self> {
                // Build path from root to self.
                let mut path = Vec::new();
                let mut cur = self;
                loop {
                    path.push(cur);
                    match cur.parent() {
                        Some(p) => cur = p,
                        None => break,
                    }
                }
                path.reverse();

                let mut result = Vec::with_capacity(1 + 2 * (path.len() - 1));
                result.push(path[0]);
                for ancestor in &path[..path.len() - 1] {
                    result.push(ancestor.left());
                    result.push(ancestor.right());
                }
                result
            }

            /// Like `subdivision_ids`, but returns only the nodes that would be leaves.
            ///
            /// These are: the target node itself, plus the sibling at every
            /// interior level (the child *not* on the path from root to target).
            /// For a root triangle, returns just itself.
            #[must_use]
            pub fn subdivision_leaf_ids(self) -> Vec<Self> {
                let depth = self.depth();
                let mut result = Vec::with_capacity(depth as usize);

                // Walk top-down from the root triangle toward self,
                // collecting the off-path sibling at each split.
                let mut ancestor = Self(0b10 | ((self.0 >> (depth - 1)) & 1));

                for i in (0..depth - 1).rev() {
                    let bit = (self.0 >> i) & 1;
                    if bit == 0 {
                        result.push(ancestor.right());
                    } else {
                        result.push(ancestor.left());
                    }
                    ancestor = if bit == 0 {
                        ancestor.left()
                    } else {
                        ancestor.right()
                    };
                }

                // The target itself is always a leaf.
                result.push(self);
                result
            }

            /// Given a set of target nodes, compute the leaves that tile the full
            /// square when every target's subdivision path is overlaid.
            ///
            /// A node is a leaf if neither of its children appear in the combined
            /// set of subdivision nodes.
            #[must_use]
            pub fn merged_leaves(targets: &[Self]) -> Vec<Self> {
                let mut all_nodes: HashSet<Self> = HashSet::new();
                all_nodes.insert(Self::TRI_0);
                all_nodes.insert(Self::TRI_1);

                for &target in targets {
                    for id in target.subdivision_ids() {
                        all_nodes.insert(id);
                    }
                }

                all_nodes
                    .iter()
                    .copied()
                    .filter(|id| id.depth() >= $MaxDepth || !all_nodes.contains(&id.left()))
                    .collect()
            }
        }
    };
}

impl_node_id!(NodeId, u32, MAX_DEPTH);
#[cfg(feature = "node64")]
impl_node_id!(NodeId64, u64, MAX_DEPTH_64);

/// Abstraction over [`NodeId`] / `NodeId64` (the latter behind the `node64`
/// feature) so [`ChunkTriTree`] can be keyed on either. `ChunkTriTree`
/// defaults to `NodeId`. The methods mirror the inherent API generated by
/// `impl_node_id!`; impls below just delegate.
pub trait NodeIdLike: Copy + Eq + core::hash::Hash + core::fmt::Debug + 'static {
    const TRI_0: Self;
    const TRI_1: Self;
    const MAX_DEPTH: u32;

    fn depth(self) -> u32;
    fn parent(self) -> Option<Self>;
    #[must_use]
    fn left(self) -> Self;
    #[must_use]
    fn right(self) -> Self;
    fn children(self) -> [Self; 2];
    fn is_left(self) -> bool;
    fn root_triangle(self) -> u32;
    fn path_bits(self) -> impl Iterator<Item = u32>;
    fn subdivision_ids(self) -> Vec<Self>;
    fn subdivision_leaf_ids(self) -> Vec<Self>;
}

macro_rules! impl_node_id_like {
    ($Name:ident, $MaxDepth:ident) => {
        impl NodeIdLike for $Name {
            const TRI_0: Self = Self::TRI_0;
            const TRI_1: Self = Self::TRI_1;
            const MAX_DEPTH: u32 = $MaxDepth;

            fn depth(self) -> u32 {
                Self::depth(self)
            }
            fn parent(self) -> Option<Self> {
                Self::parent(self)
            }
            fn left(self) -> Self {
                Self::left(self)
            }
            fn right(self) -> Self {
                Self::right(self)
            }
            fn children(self) -> [Self; 2] {
                Self::children(self)
            }
            fn is_left(self) -> bool {
                Self::is_left(self)
            }
            fn root_triangle(self) -> u32 {
                Self::root_triangle(self)
            }
            fn path_bits(self) -> impl Iterator<Item = u32> {
                Self::path_bits(self)
            }
            fn subdivision_ids(self) -> Vec<Self> {
                Self::subdivision_ids(self)
            }
            fn subdivision_leaf_ids(self) -> Vec<Self> {
                Self::subdivision_leaf_ids(self)
            }
        }
    };
}

impl_node_id_like!(NodeId, MAX_DEPTH);
#[cfg(feature = "node64")]
impl_node_id_like!(NodeId64, MAX_DEPTH_64);

/// Which diagonal of the root square is used to split it into two triangles.
///
/// ```text
///  d --- c        d --- c
///  |  /  |        |  \  |
///  a --- b        a --- b
///   AC               BD
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Diagonal {
    /// `a–c` diagonal: tri 0 = `[a, b, c]`, tri 1 = `[a, c, d]`.
    AC,
    /// `b–d` diagonal: tri 0 = `[a, b, d]`, tri 1 = `[b, c, d]`.
    BD,
}

/// Index into a [`ChunkTriTree`]'s shared vertex pool.
///
/// Multiple [`ChunkTriNode`]s reference the same `VertexId` whenever
/// their triangles meet at the same point — the position, UV and
/// normal live in a single slot, so editing one corner propagates
/// automatically to every triangle that touches it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VertexId(pub u32);

/// Per-vertex payload stored in the chunk tree's shared pool.
#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    pub position: Vec3,
    pub uv: Vec2,
    pub normal: Vec3,
}

/// Per-node payload stored in the chunk tree.
///
/// Each node holds three [`VertexId`]s into its tree's vertex pool
/// rather than owning copies of position/UV/normal. Subdivision and
/// merging therefore *never* duplicate vertex data: children of a
/// split share their non-midpoint corners with the parent's slots,
/// and merging drops the midpoint while keeping the parent's three
/// existing slots untouched.
#[derive(Clone, Copy, Debug)]
pub struct ChunkTriNode {
    /// Vertex slots in the owning tree's pool, in the leaf's stored
    /// (winding-correct) order.
    pub vertices: [VertexId; 3],
}

/// Quantized XZ key used to dedupe shared vertices in the pool.
///
/// Splits compute midpoints as `(bl + br) * 0.5` from the parent's
/// stored corner positions, and the same arithmetic on either side
/// of an RTIN diamond produces bit-identical f32 results — so a
/// to-bits key is sufficient. We canonicalise `-0.0` to `0.0` so the
/// two zero encodings hash to the same slot.
type XzKey = (u32, u32);

#[inline]
const fn key_component(x: f32) -> u32 {
    if x == 0.0 { 0 } else { x.to_bits() }
}

#[inline]
const fn xz_key(p: Vec3) -> XzKey {
    (key_component(p.x), key_component(p.z))
}

/// Flat-shading normal for a triangle (normalised cross product of two edges).
fn flat_normal(vertices: [Vec3; 3]) -> Vec3 {
    let [a, b, c] = vertices;
    (b - a).cross(c - a).normalize_or(Vec3::Y)
}

/// Sparse binary tree of subdivided triangles for a single terrain chunk.
///
/// The root square corresponds to the chunk boundary. Only materialised
/// nodes are stored; the bit-path `NodeId` encodes the tree topology so
/// no explicit child/parent pointers are needed.
///
/// Vertex data lives in a single shared pool (one slot per unique
/// XZ-position). Each [`ChunkTriNode`] holds three [`VertexId`]s into
/// that pool, and reference counts track how many nodes still touch
/// each slot. As a result:
///
/// - splitting a leaf creates exactly **one** new vertex (the base
///   midpoint, possibly already present via an RTIN diamond
///   neighbour); both children reuse the parent's three corner slots.
/// - merging back drops the midpoint slot once both children release
///   it, while the parent's three corners are preserved across the
///   round-trip.
/// - editing a corner via [`Self::vertex_mut`] is visible to every
///   triangle that touches it without any propagation step.
///
/// Internally the tree is split across two private maps so that leaf
/// iteration is `O(L)` rather than `O(L + interior_nodes)`. They are
/// only ever mutated together via the `set_leaf` / `set_interior`
/// gates and the `split` / `merge_flat` topology operations — every
/// operation that promotes or demotes a node flows through one of
/// those, so the "exactly one of leaf-with-data or interior"
/// invariant and the vertex refcounts are preserved by construction.
/// There is deliberately no public `remove`: the only way a node
/// leaves the tree is via `merge_flat`, which requires both children
/// to be leaves and promotes the parent to a leaf in the same step.
#[derive(Clone, Debug)]
pub struct ChunkTriTree<N: NodeIdLike = NodeId> {
    /// The four corners of the chunk: `[a, b, c, d]`.
    pub square: [Vec3; 4],
    /// Which diagonal splits the chunk square into two triangles.
    pub diagonal: Diagonal,
    /// Materialised leaf nodes, keyed by `NodeId`.
    leaf_nodes: HashMap<N, ChunkTriNode>,
    /// Nodes that exist in the tree but are not leaves (their children
    /// are). No payload — descent through them happens by `NodeId`
    /// arithmetic and the lookup of either child in `leaf_nodes`.
    interior_nodes: HashSet<N>,
    /// Shared vertex pool. A slot is "live" iff `refcount[i] > 0`.
    /// Freed slots remain in the vec but are returned to `free_slots`
    /// for reuse, so [`VertexId`]s stay stable across edits as long
    /// as their referrers hold them.
    vertex_pool: Vec<Vertex>,
    /// Reference counts in lock-step with `vertex_pool`. An entry of
    /// `0` means the slot is currently free.
    refcount: Vec<u32>,
    /// Reusable vertex slot indices, populated when a slot's refcount
    /// drops to zero.
    free_slots: Vec<u32>,
    /// Quantized-XZ → `VertexId` map used by [`Self::intern_vertex`]
    /// to dedupe vertices that meet at the same chunk-local position.
    by_xz: HashMap<XzKey, VertexId>,
}

impl<N: NodeIdLike> ChunkTriTree<N> {
    /// Create a chunk tree from four corner positions and a diagonal orientation.
    ///
    /// Stored vertex order is CCW from above (positive-Y face normal) for
    /// callers that pass corners CCW around the square in XZ:
    /// `Diagonal::AC` → tri 0 = `[a, c, b]`, tri 1 = `[a, d, c]`
    /// `Diagonal::BD` → tri 0 = `[a, d, b]`, tri 1 = `[b, d, c]`
    #[must_use]
    pub fn new(square: [Vec3; 4], diagonal: Diagonal) -> Self {
        let mut tree = Self {
            square,
            diagonal,
            leaf_nodes: HashMap::new(),
            interior_nodes: HashSet::new(),
            vertex_pool: Vec::new(),
            refcount: Vec::new(),
            free_slots: Vec::new(),
            by_xz: HashMap::new(),
        };

        let [a, b, c, d] = square;
        let (tri0, tri1) = match diagonal {
            Diagonal::AC => ([a, c, b], [a, d, c]),
            Diagonal::BD => ([a, d, b], [b, d, c]),
        };

        let n0 = flat_normal(tri0);
        let n1 = flat_normal(tri1);
        let ids0 = tri0.map(|v| tree.intern_vertex(v, Vec2::new(v.x, v.z), n0));
        let ids1 = tri1.map(|v| tree.intern_vertex(v, Vec2::new(v.x, v.z), n1));

        tree.set_leaf(N::TRI_0, ChunkTriNode { vertices: ids0 });
        tree.set_leaf(N::TRI_1, ChunkTriNode { vertices: ids1 });
        tree
    }

    // ── Vertex pool ─────────────────────────────────────────────────

    /// Borrow a vertex by id.
    ///
    /// # Panics
    /// Panics if `id` does not refer to a live slot.
    #[must_use]
    pub fn vertex(&self, id: VertexId) -> &Vertex {
        debug_assert!(
            self.refcount[id.0 as usize] > 0,
            "VertexId {id:?} is not live",
        );
        &self.vertex_pool[id.0 as usize]
    }

    /// Mutably borrow a vertex by id. Edits are visible to every
    /// [`ChunkTriNode`] that references this slot.
    ///
    /// # Panics
    /// Panics if `id` does not refer to a live slot.
    #[must_use]
    pub fn vertex_mut(&mut self, id: VertexId) -> &mut Vertex {
        debug_assert!(
            self.refcount[id.0 as usize] > 0,
            "VertexId {id:?} is not live",
        );
        &mut self.vertex_pool[id.0 as usize]
    }

    /// Iterate over every live vertex slot, paired with its id.
    pub fn vertices(&self) -> impl Iterator<Item = (VertexId, &Vertex)> {
        #[allow(clippy::cast_possible_truncation)]
        self.vertex_pool
            .iter()
            .enumerate()
            .filter(|&(i, _)| self.refcount[i] > 0)
            .map(|(i, v)| (VertexId(i as u32), v))
    }

    /// Iterate mutably over every live vertex slot.
    ///
    /// Each unique vertex is visited exactly once regardless of how
    /// many leaves reference it — the natural way to apply a brush
    /// to terrain without double-counting shared corners.
    pub fn vertices_mut(&mut self) -> impl Iterator<Item = (VertexId, &mut Vertex)> {
        self.vertex_pool
            .iter_mut()
            .zip(self.refcount.iter())
            .enumerate()
            .filter_map(|(i, (v, rc))| {
                #[allow(clippy::cast_possible_truncation)]
                (*rc > 0).then_some((VertexId(i as u32), v))
            })
    }

    /// Number of live vertex slots in the pool.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.refcount.iter().filter(|&&r| r > 0).count()
    }

    /// Intern a vertex at `position`, returning an existing slot if a
    /// vertex with the same XZ key is already present. The first
    /// caller's UV / normal data wins; subsequent callers reuse the
    /// existing slot's data unchanged.
    fn intern_vertex(&mut self, position: Vec3, uv: Vec2, normal: Vec3) -> VertexId {
        let key = xz_key(position);
        if let Some(&id) = self.by_xz.get(&key) {
            return id;
        }
        let id = if let Some(slot) = self.free_slots.pop() {
            debug_assert_eq!(
                self.refcount[slot as usize], 0,
                "free_slots holds slot {slot} with non-zero refcount — \
                 a live vertex would be silently overwritten",
            );
            self.vertex_pool[slot as usize] = Vertex {
                position,
                uv,
                normal,
            };
            VertexId(slot)
        } else {
            #[allow(clippy::cast_possible_truncation)]
            let slot = self.vertex_pool.len() as u32;
            self.vertex_pool.push(Vertex {
                position,
                uv,
                normal,
            });
            self.refcount.push(0);
            VertexId(slot)
        };
        self.by_xz.insert(key, id);
        id
    }

    fn retain(&mut self, id: VertexId) {
        self.refcount[id.0 as usize] += 1;
    }

    fn release(&mut self, id: VertexId) {
        let i = id.0 as usize;
        debug_assert!(self.refcount[i] > 0);
        self.refcount[i] -= 1;
        if self.refcount[i] == 0 {
            let key = xz_key(self.vertex_pool[i].position);
            self.by_xz.remove(&key);
            self.free_slots.push(id.0);
        }
    }

    // ── Cell access (the only gated mutators) ───────────────────────

    /// Borrow the leaf node at `id`, if any. Returns `None` for
    /// `interior_nodes` and unknown ids.
    #[must_use]
    pub fn get(&self, id: &N) -> Option<&ChunkTriNode> {
        self.leaf_nodes.get(id)
    }

    /// Position triple for the leaf at `id`, if any.
    #[must_use]
    pub fn leaf_positions(&self, id: N) -> Option<[Vec3; 3]> {
        let node = self.leaf_nodes.get(&id)?;
        Some(node.vertices.map(|v| self.vertex(v).position))
    }

    /// UV triple for the leaf at `id`, if any.
    #[must_use]
    pub fn leaf_uvs(&self, id: N) -> Option<[Vec2; 3]> {
        let node = self.leaf_nodes.get(&id)?;
        Some(node.vertices.map(|v| self.vertex(v).uv))
    }

    /// Normal triple for the leaf at `id`, if any.
    #[must_use]
    pub fn leaf_normals(&self, id: N) -> Option<[Vec3; 3]> {
        let node = self.leaf_nodes.get(&id)?;
        Some(node.vertices.map(|v| self.vertex(v).normal))
    }

    /// Combined `(positions, uvs, normals)` snapshot for the leaf at
    /// `id`, if any. Convenient for mesh-building snapshots.
    #[must_use]
    pub fn leaf_data(&self, id: N) -> Option<([Vec3; 3], [Vec2; 3], [Vec3; 3])> {
        let node = self.leaf_nodes.get(&id)?;
        let v: [&Vertex; 3] = node.vertices.map(|v| self.vertex(v));
        Some((
            [v[0].position, v[1].position, v[2].position],
            [v[0].uv, v[1].uv, v[2].uv],
            [v[0].normal, v[1].normal, v[2].normal],
        ))
    }

    /// Install `id` as a leaf with the supplied vertex slot ids. If
    /// `id` was previously a leaf, its old vertex references are
    /// released first; if it was interior, that membership is dropped.
    /// Refcounts on `node`'s ids are bumped exactly once.
    fn set_leaf(&mut self, id: N, node: ChunkTriNode) {
        // Retain new vertex refs *before* releasing the previous leaf's,
        // so a slot shared between prev and node never dips to rc=0. If
        // it did, `release` would push the slot onto `free_slots` and
        // the subsequent `retain` would leave it live (rc=1) but still
        // queued for reuse — the next `intern_vertex` would then pop
        // the slot and silently clobber an in-use vertex.
        for v in node.vertices {
            self.retain(v);
        }
        if let Some(prev) = self.leaf_nodes.remove(&id) {
            for v in prev.vertices {
                self.release(v);
            }
        } else {
            self.interior_nodes.remove(&id);
        }
        self.leaf_nodes.insert(id, node);
    }

    /// Mark `id` as interior. If `id` was previously a leaf, its
    /// vertex references are released.
    fn set_interior(&mut self, id: N) {
        if let Some(prev) = self.leaf_nodes.remove(&id) {
            for v in prev.vertices {
                self.release(v);
            }
        }
        self.interior_nodes.insert(id);
    }

    /// Drop `id` from the tree entirely (must currently be a leaf),
    /// releasing its vertex references. Used by `merge_flat` to
    /// dissolve children once the parent has absorbed their data.
    fn remove_leaf(&mut self, id: N) {
        if let Some(prev) = self.leaf_nodes.remove(&id) {
            for v in prev.vertices {
                self.release(v);
            }
        }
    }

    /// Escape hatch: install `id` as a leaf carrying a triangle with
    /// `positions`, `uvs` and `normals`, bypassing the topology
    /// invariants enforced by `split` / `merge_flat`.
    ///
    /// Vertices are interned so corners shared with already-present
    /// leaves coalesce automatically. UV and normal are written
    /// last-caller-wins on every call — `intern_vertex` is first-caller-
    /// wins, but the root corners pre-seeded by [`Self::new`] carry
    /// placeholder UV/normal that must be overwritten by the actual
    /// caller-provided attributes. Intended for bottom-up tree
    /// construction (e.g. mesh import), where the caller is responsible
    /// for also marking every ancestor as interior via
    /// [`Self::force_set_interior`] so descent queries terminate at the
    /// right level. Prefer [`Self::split`] for normal use.
    pub fn force_set_leaf(
        &mut self,
        id: N,
        positions: [Vec3; 3],
        uvs: [Vec2; 3],
        normals: [Vec3; 3],
    ) {
        let vertices = core::array::from_fn(|i| {
            let vid = self.intern_vertex(positions[i], uvs[i], normals[i]);
            let slot = &mut self.vertex_pool[vid.0 as usize];
            slot.uv = uvs[i];
            slot.normal = normals[i];
            vid
        });
        self.set_leaf(id, ChunkTriNode { vertices });
    }

    /// Escape hatch: mark `id` as interior (no payload), bypassing
    /// the topology invariants enforced by `split` / `merge_flat`.
    ///
    /// Intended for bottom-up tree construction (e.g. mesh import),
    /// where every ancestor of an imported leaf must be marked
    /// interior so descent queries don't stop early. Prefer
    /// [`Self::split`] for normal use.
    pub fn force_set_interior(&mut self, id: N) {
        self.set_interior(id);
    }

    /// Whether any slot (leaf or interior) exists at `id`.
    #[must_use]
    pub fn contains(&self, id: &N) -> bool {
        self.leaf_nodes.contains_key(id) || self.interior_nodes.contains(id)
    }

    #[must_use]
    fn has_children(&self, id: N) -> bool {
        id.depth() < N::MAX_DEPTH && self.interior_nodes.contains(&id)
    }

    /// Whether this node exists and has no children in the tree.
    #[must_use]
    pub fn is_leaf(&self, id: N) -> bool {
        self.leaf_nodes.contains_key(&id)
    }

    /// Iterate all leaf nodes. `O(L)` — walks only the leaf map.
    pub fn leaves(&self) -> impl Iterator<Item = (&N, &ChunkTriNode)> {
        self.leaf_nodes.iter()
    }
    // Iterate all leaf nodes mutably. `O(L)`.
    // UNUSED
    // pub fn leaves_mut(&mut self) -> impl Iterator<Item = (&N, &mut ChunkTriNode)> {
    //     self.leaf_nodes.iter_mut()
    // }

    /// Total number of materialised leaf nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.leaf_nodes.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.leaf_nodes.is_empty()
    }

    /// Side length of the chunk square (distance `a → b`).
    #[must_use]
    pub fn chunk_side(&self) -> f32 {
        self.square[0].distance(self.square[1])
    }

    /// Leg length of a right-isosceles triangle at the given subdivision level.
    ///
    /// In this terrain representation, each pair of base-edge-neighbour
    /// triangles (RTIN diamond pair, sharing their hypotenuse) forms a
    /// square cell whose side equals this leg length.
    /// Level 1 = the two root triangles (leg = chunk side).
    /// Each subsequent level shrinks by `1/√2`.
    #[must_use]
    pub fn leg_at_level(&self, level: u32) -> f32 {
        self.chunk_side() / fmath::sqrt(fmath::powi(2.0_f32, level.cast_signed() - 1))
    }

    /// Base (longest edge) length at the given subdivision level (`leg × √2`).
    #[must_use]
    pub fn base_at_level(&self, level: u32) -> f32 {
        self.leg_at_level(level) * core::f32::consts::SQRT_2
    }

    /// Subdivision level whose fine square edge length is closest to
    /// `target_square_edge`.
    ///
    /// Clamps to a minimum of 1 (root triangles).
    #[must_use]
    pub fn level_for_square_edge(&self, target_square_edge: f32) -> u32 {
        // leg = side / sqrt(2)^(n-1)  ⟹  n = 1 + 2·log₂(side / leg)
        let n = fmath::mul_add(
            2.0f32,
            fmath::log2(self.chunk_side() / target_square_edge),
            1.0,
        );
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        {
            (fmath::round(n) as u32).max(1)
        }
    }

    /// Subdivision level whose base (longest edge) is closest to `target_base`.
    ///
    /// Clamps to a minimum of 1 (root triangles).
    #[must_use]
    pub fn level_for_base(&self, target_base: f32) -> u32 {
        self.level_for_square_edge(target_base / core::f32::consts::SQRT_2)
    }

    /// Root-triangle vertices as `[apex, base_left, base_right]`.
    ///
    /// Apex is opposite the base (the diagonal). The base is split on
    /// each subdivision step.
    const fn root_tri_vertices(&self, root_tri: u32) -> (Vec3, Vec3, Vec3) {
        let [a, b, c, d] = self.square;
        match (self.diagonal, root_tri) {
            (Diagonal::AC, 0) => (b, a, c),
            (Diagonal::AC, _) => (d, c, a),
            (Diagonal::BD, 0) => (a, b, d),
            (Diagonal::BD, _) => (c, d, b),
        }
    }

    /// Compute flat `[apex, base_left, base_right]` for any `NodeId`, whether or
    /// not it is materialised in the tree.
    ///
    /// Each split bisects the base at its midpoint `M`. The children are:
    /// - left  (bit 0): `[M, base_left, old_apex]`
    /// - right (bit 1): `[M, old_apex, base_right]`
    #[must_use]
    pub fn vertices_of(&self, id: &N) -> [Vec3; 3] {
        let (mut apex, mut base_l, mut base_r) = self.root_tri_vertices(id.root_triangle());

        for bit in id.path_bits().skip(1) {
            let mid = (base_l + base_r) * 0.5;
            let old_apex = apex;
            apex = mid;
            if bit == 0 {
                base_r = old_apex;
            } else {
                base_l = old_apex;
            }
        }

        (apex, base_l, base_r).into()
    }

    /// XZ midpoint of the base (longest edge) for the given node.
    #[must_use]
    pub fn base_midpoint_xz(&self, id: &N) -> Vec2 {
        let [_, l, r] = self.vertices_of(id);
        let m = (l + r) * 0.5;
        Vec2::new(m.x, m.z)
    }

    /// XZ centroid of the triangle for the given node.
    #[must_use]
    pub fn centroid_xz(&self, id: &N) -> Vec2 {
        let [a, b, c] = self.vertices_of(id);
        let cen = (a + b + c) / 3.0;
        Vec2::new(cen.x, cen.z)
    }

    /// Find the `NodeId` at a specific `depth` whose triangle contains `xz`.
    ///
    /// Purely geometric — ignores which nodes are materialised. Returns `None`
    /// if the point is outside the chunk square.
    #[must_use]
    pub fn id_at_xz(&self, xz: Vec2, depth: u32) -> Option<N> {
        debug_assert!(depth <= N::MAX_DEPTH);

        // Determine which root triangle the point belongs to.
        let (diag_start, diag_end) = match self.diagonal {
            Diagonal::AC => (self.square[0], self.square[2]),
            Diagonal::BD => (self.square[1], self.square[3]),
        };
        let diag = Vec2::new(diag_end.x - diag_start.x, diag_end.z - diag_start.z);
        let to_point = xz - Vec2::new(diag_start.x, diag_start.z);

        let (tri0_apex, _, _) = self.root_tri_vertices(0);
        let to_ref = Vec2::new(tri0_apex.x - diag_start.x, tri0_apex.z - diag_start.z);

        let ref_cross = cross_xz(diag, to_ref);
        let pt_cross = cross_xz(diag, to_point);

        let mut id = if ref_cross * pt_cross >= 0.0 {
            N::TRI_0
        } else {
            N::TRI_1
        };

        let (mut apex, mut base_l, mut base_r) = self.root_tri_vertices(id.root_triangle());

        // Verify the point is actually inside the root triangle (not just on
        // the correct side of the diagonal but outside the square).
        let edge_al = Vec2::new(base_l.x - apex.x, base_l.z - apex.z);
        let edge_lr = Vec2::new(base_r.x - base_l.x, base_r.z - base_l.z);
        let edge_ra = Vec2::new(apex.x - base_r.x, apex.z - base_r.z);

        let c1 = cross_xz(edge_al, xz - Vec2::new(apex.x, apex.z));
        let c2 = cross_xz(edge_lr, xz - Vec2::new(base_l.x, base_l.z));
        let c3 = cross_xz(edge_ra, xz - Vec2::new(base_r.x, base_r.z));

        if !((c1 >= 0.0 && c2 >= 0.0 && c3 >= 0.0) || (c1 <= 0.0 && c2 <= 0.0 && c3 <= 0.0)) {
            return None;
        }

        for _ in 1..depth {
            let mid = (base_l + base_r) * 0.5;
            let split_dir = Vec2::new(mid.x - apex.x, mid.z - apex.z);
            let to_bl = Vec2::new(base_l.x - apex.x, base_l.z - apex.z);
            let to_pt = xz - Vec2::new(apex.x, apex.z);

            let bl_cross = cross_xz(split_dir, to_bl);
            let pt_cross = cross_xz(split_dir, to_pt);

            let old_apex = apex;
            apex = mid;
            if bl_cross * pt_cross >= 0.0 {
                id = id.left();
                base_r = old_apex;
            } else {
                id = id.right();
                base_l = old_apex;
            }
        }

        Some(id)
    }

    /// Find the deepest materialised leaf containing `xz`.
    ///
    /// Walks down the tree following splits until it reaches a leaf (a node
    /// whose children are not in the tree). Returns `None` if the point is
    /// outside the chunk.
    #[must_use]
    pub fn leaf_at_xz(&self, xz: Vec2) -> Option<N> {
        // Start at depth 1 to select the root triangle.
        let mut id = self.id_at_xz(xz, 1)?;
        let (mut apex, mut base_l, mut base_r) = self.root_tri_vertices(id.root_triangle());

        while self.has_children(id) {
            let mid = (base_l + base_r) * 0.5;
            let split_dir = Vec2::new(mid.x - apex.x, mid.z - apex.z);
            let to_bl = Vec2::new(base_l.x - apex.x, base_l.z - apex.z);
            let to_pt = xz - Vec2::new(apex.x, apex.z);

            let bl_cross = cross_xz(split_dir, to_bl);
            let pt_cross = cross_xz(split_dir, to_pt);

            let old_apex = apex;
            apex = mid;
            let next = if bl_cross * pt_cross >= 0.0 {
                base_r = old_apex;
                id.left()
            } else {
                base_l = old_apex;
                id.right()
            };

            if !self.contains(&next) {
                break;
            }

            id = next;
        }

        Some(id)
    }

    /// Interpolated surface height at `xz`, using the deepest leaf triangle.
    ///
    /// Returns `None` if the point is outside the chunk.
    #[must_use]
    pub fn height_at_xz(&self, xz: Vec2) -> Option<f32> {
        let id = self.leaf_at_xz(xz)?;
        let [a, b, c] = self.leaf_positions(id)?;
        let (wa, wb, wc) = crate::barycentric_coords_xz(
            xz,
            Vec2::new(a.x, a.z),
            Vec2::new(b.x, b.z),
            Vec2::new(c.x, c.z),
        )?;
        Some(fmath::mul_add(wa, a.y, fmath::mul_add(wb, b.y, wc * c.y)))
    }

    /// Return every `NodeId` that was subdivided into existence on the path
    /// from the root triangle down to the deepest materialised leaf
    /// containing `xz`.
    ///
    /// Each split creates both children, so the result includes the root
    /// triangle, both children at every interior level, and the leaf.
    /// Returns `None` if the point is outside the chunk.
    #[must_use]
    pub fn subdivision_ids_to_leaf_at_xz(&self, xz: Vec2) -> Option<Vec<N>> {
        let mut id = self.id_at_xz(xz, 1)?;
        let (mut apex, mut base_l, mut base_r) = self.root_tri_vertices(id.root_triangle());

        let mut ids = vec![id];

        while self.has_children(id) {
            let mid = (base_l + base_r) * 0.5;
            let split_dir = Vec2::new(mid.x - apex.x, mid.z - apex.z);
            let to_bl = Vec2::new(base_l.x - apex.x, base_l.z - apex.z);
            let to_pt = xz - Vec2::new(apex.x, apex.z);

            let bl_cross = cross_xz(split_dir, to_bl);
            let pt_cross = cross_xz(split_dir, to_pt);

            let old_apex = apex;
            apex = mid;
            let left = id.left();
            let right = id.right();
            if self.contains(&left) {
                ids.push(left);
            }
            if self.contains(&right) {
                ids.push(right);
            }

            let next = if bl_cross * pt_cross >= 0.0 {
                base_r = old_apex;
                left
            } else {
                base_l = old_apex;
                right
            };

            if !self.contains(&next) {
                break;
            }

            id = next;
        }

        Some(ids)
    }

    // ── Subdivision ─────────────────────────────────────────────────

    /// Split a leaf node into two children. The two non-midpoint
    /// corners of each child are *aliased* to the parent's existing
    /// vertex slots — no copying. The base midpoint is interned, so
    /// it coalesces automatically with an already-split RTIN diamond
    /// neighbour.
    ///
    /// Returns `None` if the node doesn't exist, is already split, or
    /// is at maximum depth.
    #[allow(clippy::similar_names)]
    pub fn split(&mut self, parent_id: &N) -> Option<(N, N)> {
        if parent_id.depth() >= N::MAX_DEPTH || self.has_children(*parent_id) {
            return None;
        }
        let parent = *self.leaf_nodes.get(parent_id)?;

        let left_id = parent_id.left();
        let right_id = parent_id.right();

        // Locate the parent's apex / base_left / base_right slots so
        // we can route them into the children's correct roles.
        let [geo_apex, geo_bl, geo_br] = self.vertices_of(parent_id);
        let parent_positions = parent.vertices.map(|v| self.vertex(v).position);
        let ai = nearest_vertex_xz(&parent_positions, geo_apex);
        let bli = nearest_vertex_xz(&parent_positions, geo_bl);
        let bri = nearest_vertex_xz(&parent_positions, geo_br);
        let apex_id = parent.vertices[ai];
        let bl_id = parent.vertices[bli];
        let br_id = parent.vertices[bri];

        // Compute the midpoint vertex (or reuse an existing one if a
        // diamond neighbour already split). UV / normal seeds are
        // discarded by `intern_vertex` if the slot already exists.
        let bl = self.vertex(bl_id);
        let br = self.vertex(br_id);
        let mid_pos = (bl.position + br.position) * 0.5;
        let mid_uv = (bl.uv + br.uv) * 0.5;
        let mid_normal = ((bl.normal + br.normal) * 0.5).normalize_or(Vec3::Y);
        let mid_id = self.intern_vertex(mid_pos, mid_uv, mid_normal);

        // Parent winding sign (Y component of face normal).
        let parent_sign = (parent_positions[1] - parent_positions[0])
            .cross(parent_positions[2] - parent_positions[0])
            .y;

        // Build children in canonical RTIN order, then swap slots 1 & 2
        // if the resulting winding doesn't match the parent. The
        // canonical split alternates winding at each depth, but mesh
        // imports preserve incoming winding — so all leaves of the
        // tree share a sign and we keep that consistent here.
        //
        // Positions are looked up via `parent_positions` / `mid_pos`
        // rather than through `self.vertex(...)` because `mid_id` may
        // be a freshly interned slot whose refcount is still zero
        // (it's only bumped below by `set_leaf` → `retain`).
        let apex_pos = parent_positions[ai];
        let bl_pos = parent_positions[bli];
        let br_pos = parent_positions[bri];

        let mut left_ids = [mid_id, bl_id, apex_id];
        let lp = [mid_pos, bl_pos, apex_pos];
        let left_sign = (lp[1] - lp[0]).cross(lp[2] - lp[0]).y;
        if left_sign * parent_sign < 0.0 {
            left_ids.swap(1, 2);
        }

        let mut right_ids = [mid_id, apex_id, br_id];
        let rp = [mid_pos, apex_pos, br_pos];
        let right_sign = (rp[1] - rp[0]).cross(rp[2] - rp[0]).y;
        if right_sign * parent_sign < 0.0 {
            right_ids.swap(1, 2);
        }

        // Atomically install children first (so their `retain`s bump
        // shared corners up before the parent's `release` would
        // otherwise drop them to zero and free their slots), then demote
        // the parent. All three updates flow through the gated mutators
        // so the leaves/interior_nodes split and the vertex refcounts
        // stay consistent.
        self.set_leaf(left_id, ChunkTriNode { vertices: left_ids });
        self.set_leaf(
            right_id,
            ChunkTriNode {
                vertices: right_ids,
            },
        );
        self.set_interior(*parent_id);

        Some((left_id, right_id))
    }

    /// Return the geometric base-edge neighbor of `id` at the same depth.
    ///
    /// In RTIN, two right-isosceles triangles that share their hypotenuse
    /// (longest edge, the "base") form a diamond. This returns the diamond
    /// pair as a `NodeId` at the same depth as `id`, computed purely from
    /// geometry — the returned id may or may not be materialised in the
    /// tree.
    ///
    /// Returns `None` when the base edge lies on the chunk boundary, so
    /// the would-be neighbor is in a different chunk (or off the world).
    #[must_use]
    pub fn base_edge_neighbor(&self, id: &N) -> Option<N> {
        let [apex, bl, br] = self.vertices_of(id);
        let mid_xz = Vec2::new((bl.x + br.x) * 0.5, (bl.z + br.z) * 0.5);
        let apex_xz = Vec2::new(apex.x, apex.z);
        let dir = mid_xz - apex_xz;
        let dir_len = dir.length();
        if dir_len <= 0.0 {
            return None;
        }
        // Step a tiny way past the base edge, away from `apex`, into the
        // adjacent triangle. The chunk's diagonal sets a length scale so
        // the offset is small relative to the geometry but well above
        // f32 noise from the cross-product side tests in `id_at_xz`.
        let chunk_diag = fmath::max((self.square[2] - self.square[0]).length(), 1.0);
        let eps = chunk_diag * 1e-4;
        let probe = mid_xz + dir * (eps / dir_len);
        let n = self.id_at_xz(probe, id.depth())?;
        // If the probe didn't actually cross the edge (e.g. degenerate
        // triangle), there's no meaningful neighbor.
        (n != *id).then_some(n)
    }

    /// Split `id` while maintaining the RTIN diamond rule: the base-edge
    /// neighbor (the triangle across the longest edge) is also split,
    /// recursively bringing coarser neighbors up to the same depth first.
    ///
    /// This avoids T-junctions between subdivided and non-subdivided
    /// regions. Because the base midpoint is interned in the shared
    /// vertex pool, both diamond halves end up referencing the *same*
    /// midpoint slot — a subsequent per-vertex displacement (e.g. brush
    /// sculpting) automatically stays seam-free across the diamond.
    ///
    /// Cross-chunk neighbors are not handled here (the chunk's tree only
    /// knows about its own square). The caller should use the same world-
    /// aligned target depth across chunks so cross-chunk edges agree by
    /// construction.
    ///
    /// Returns `None` if `id` doesn't exist or is at maximum depth.
    pub fn symmetric_split(&mut self, parent_id: &N) -> Option<(N, N)> {
        if parent_id.depth() >= N::MAX_DEPTH || !self.is_leaf(*parent_id) {
            return None;
        }

        if self.has_children(*parent_id) {
            return Some((parent_id.left(), parent_id.right()));
        }

        let Some(neighbor) = self.base_edge_neighbor(parent_id) else {
            // Boundary case: base edge is on the chunk square. Just split.
            return self.split(parent_id);
        };

        if !self.is_leaf(neighbor) {
            // Geometric neighbor isn't materialised yet. Walk up to its
            // deepest existing ancestor and force-split that to bring
            // the neighbor side one level closer to our depth.
            let Some(mut anc) = neighbor.parent() else {
                return self.split(parent_id);
            };
            while !self.is_leaf(anc) {
                match anc.parent() {
                    Some(p) => anc = p,
                    None => return self.split(parent_id),
                }
            }
            // Recursing on the ancestor splits it (and propagates the
            // diamond rule on its side too). Then re-enter to handle
            // `id`; eventually `neighbor` exists at our depth.
            self.symmetric_split(&anc);
            return self.symmetric_split(parent_id);
        }

        // Neighbor exists at our depth. Split both to keep the diamond
        // closed. If the neighbor is somehow already split (e.g. from a
        // prior cascade), splitting `id` alone is still correct — the
        // shared edge is already bisected on its side.
        let split_self = self.split(parent_id);
        if !self.has_children(neighbor) {
            self.split(&neighbor);
        }
        split_self
    }

    // ── Merging ─────────────────────────────────────────────────────

    /// Merge a node's two child leaves back into the parent if the
    /// midpoint shared by the children is approximately colinear with
    /// the parent's base edge — i.e. the child pair doesn't carry any
    /// height detail beyond the planar interpolation of the parent.
    ///
    /// Returns `true` if the merge happened. Fails if either child is
    /// missing, either child is itself split, or the midpoint's Y
    /// deviates from the linear interpolation between the base
    /// endpoints by more than `y_tolerance`.
    ///
    /// Because the parent's three corner vertex slots were never
    /// duplicated when the original split happened (children alias
    /// them in the shared pool), promoting the parent back to a leaf
    /// is just a topology flip — any height edits applied to the
    /// children's apex / bl / br corners are *already* visible on the
    /// merged leaf. Only the midpoint slot is dropped.
    #[allow(clippy::similar_names)]
    pub fn merge_flat(&mut self, id: N, y_tolerance: f32) -> bool {
        // The parent of two leaf children must itself be interior;
        // anything else means this id isn't a valid merge target.
        if !self.interior_nodes.contains(&id) {
            return false;
        }

        let left = id.left();
        let Some(left_node) = self.leaf_nodes.get(&left) else {
            return false;
        };

        let right = id.right();
        let Some(right_node) = self.leaf_nodes.get(&right) else {
            return false;
        };

        // `canonical_indices_for_child` returns slots in the child's
        // `[apex, base_left, base_right]` order. For the *left* child,
        // base_left = parent's bl and base_right = parent's old apex.
        // For the *right* child, base_left = parent's old apex and
        // base_right = parent's br — so the parent-role labels swap.
        let [l_mid_i, l_bl_i, l_ap_i] = canonical_indices_for_child(self, left, left_node.vertices);
        let [r_mid_i, _discarded_r_ap_i, r_br_i] =
            canonical_indices_for_child(self, right, right_node.vertices);

        let l_mid_pos = self.vertex(left_node.vertices[l_mid_i]).position;
        let r_mid_pos = self.vertex(right_node.vertices[r_mid_i]).position;
        let l_bl_pos = self.vertex(left_node.vertices[l_bl_i]).position;
        let r_br_pos = self.vertex(right_node.vertices[r_br_i]).position;

        let mid_y = (l_mid_pos.y + r_mid_pos.y) * 0.5;
        let expected_mid_y = (l_bl_pos.y + r_br_pos.y) * 0.5;
        if fmath::abs(mid_y - expected_mid_y) > y_tolerance {
            return false;
        }

        // Both children's apex slots are the same shared VertexId
        // (the parent's old apex), so picking either is fine.
        let apex_id = left_node.vertices[l_ap_i];
        let bl_id = left_node.vertices[l_bl_i];
        let br_id = right_node.vertices[r_br_i];

        // Build the merged triangle with winding matching the children
        // (rather than the canonical [apex, bl, br] order), so the
        // merged leaf survives backface culling and stays visible.
        let lp = left_node.vertices.map(|v| self.vertex(v).position);
        let child_sign = (lp[1] - lp[0]).cross(lp[2] - lp[0]).y;

        let mut merged_ids = [apex_id, bl_id, br_id];
        let mp = merged_ids.map(|v| self.vertex(v).position);
        let merged_sign = (mp[1] - mp[0]).cross(mp[2] - mp[0]).y;
        if merged_sign * child_sign < 0.0 {
            merged_ids.swap(1, 2);
        }

        // Atomically promote the parent's slot from interior to leaf
        // with the children's surviving corner ids, and drop both
        // children. The midpoint VertexId loses both its referrers
        // and is returned to the free list inside `remove_leaf`.
        self.set_leaf(
            id,
            ChunkTriNode {
                vertices: merged_ids,
            },
        );
        self.remove_leaf(left);
        self.remove_leaf(right);

        // Snap any T-junction vertex that still sits on the new base edge.
        // If the base-edge diamond partner is still subdivided, its children
        // keep M alive with their shared apex at exactly mid(bl, br). That
        // apex lies on our now-straight edge [bl, br] — a T-junction. Since
        // the merge was accepted (|mid_y - expected_mid_y| <= tolerance), we
        // snap M.y onto the flat plane to make the crack geometrically zero.
        // `remove_leaf` already dropped P's children's refs: if M's refcount
        // fell to zero the diamond partner wasn't split and there is no
        // T-junction — `by_xz` will simply return None.
        let mid_key = xz_key(Vec3::new(
            (l_bl_pos.x + r_br_pos.x) * 0.5,
            0.0,
            (l_bl_pos.z + r_br_pos.z) * 0.5,
        ));
        if let Some(&mid_id) = self.by_xz.get(&mid_key) {
            self.vertex_pool[mid_id.0 as usize].position.y = expected_mid_y;
        }

        true
    }

    /// Repeatedly merge every flat leaf pair (children whose shared
    /// midpoint is within `y_tolerance` of the planar interpolation of
    /// their parent's base edge) until no more merges are possible.
    ///
    /// Works bottom-up: once a pair is merged, the resulting leaf may
    /// itself become part of another mergeable pair at the next level.
    ///
    /// Returns the total number of pairs merged.
    pub fn simplify_flat(&mut self, y_tolerance: f32) -> usize {
        let mut total = 0usize;
        loop {
            // Collect each candidate parent once per pass. A parent is
            // a candidate iff both its children are materialised leaves.
            let mut candidates: HashSet<N> = HashSet::new();
            for id in self.leaf_nodes.keys() {
                if let Some(parent) = id.parent() {
                    candidates.insert(parent);
                }
            }
            let mut merged_this_pass = 0usize;
            for parent in candidates {
                if self.merge_flat(parent, y_tolerance) {
                    merged_this_pass += 1;
                }
            }
            if merged_this_pass == 0 {
                break;
            }
            total += merged_this_pass;
        }
        total
    }

    /// Return the sequence of `NodeId`s from the current materialised leaf
    /// containing `xz` down to the geometric node at `depth`, so that each
    /// one can be `split()` in order to reach the target resolution.
    ///
    /// The returned vec starts at the leaf and ends at the target depth.
    /// If `depth` is at or above the current leaf, returns an empty vec.
    /// Returns `None` if `xz` is outside the chunk.
    #[must_use]
    pub fn path_from_leaf_to_depth(&self, xz: Vec2, depth: u32) -> Option<Vec<N>> {
        let leaf = self.leaf_at_xz(xz)?;
        let leaf_depth = leaf.depth();
        if depth <= leaf_depth {
            return Some(Vec::new());
        }
        let target = self.id_at_xz(xz, depth)?;
        // Walk from the target back up to the current leaf, then prepend the
        // leaf so callers can `split()` each node in order.
        let count = (depth - leaf_depth + 1) as usize;
        let mut descendants = Vec::with_capacity(count - 1);
        let mut cur = target;
        while cur != leaf {
            descendants.push(cur);
            cur = cur.parent()?;
        }
        let mut path = Vec::with_capacity(count);
        path.push(leaf);
        path.extend(descendants.into_iter().rev());
        Some(path)
    }

    /// Check the tree's topology invariants: every non-root leaf has
    /// an interior parent, and every interior has both children
    /// present. Intended for tests and post-import sanity checks
    /// after `force_set_*` use.
    #[must_use]
    pub fn validate(&self) -> bool {
        // Every leaf's parent must be interior, and every interior
        // must have both children in the tree.
        for leaf_id in self.leaf_nodes.keys() {
            if let Some(parent) = leaf_id.parent()
                && !self.interior_nodes.contains(&parent)
            {
                return false;
            }
        }
        for interior in &self.interior_nodes {
            let left = interior.left();
            let right = interior.right();
            if !self.contains(&left) || !self.contains(&right) {
                return false;
            }
        }
        true
    }
}

/// Identify the slot indices of a child leaf created by `split()`,
/// returning `[mid_idx, outer_idx, apex_idx]` where:
/// - `mid_idx` is the slot that lies at the parent's base midpoint,
/// - `outer_idx` is the slot at the parent's base endpoint that this
///   child kept (`bl` for the left child, `br` for the right child),
/// - `apex_idx` is the slot that holds the parent's old apex.
#[allow(clippy::similar_names)]
fn canonical_indices_for_child<N: NodeIdLike>(
    tree: &ChunkTriTree<N>,
    child_id: N,
    vertices: [VertexId; 3],
) -> [usize; 3] {
    // The child's apex (in its own coordinate frame) is the parent's
    // base midpoint; its base_left and base_right are the parent's
    // base-endpoint and old-apex (in left/right order).
    let [child_apex, child_bl, child_br] = tree.vertices_of(&child_id);
    let positions = vertices.map(|v| tree.vertex(v).position);
    let mid_idx = nearest_vertex_xz(&positions, child_apex);
    let outer_idx = nearest_vertex_xz(&positions, child_bl);
    let apex_idx = nearest_vertex_xz(&positions, child_br);
    [mid_idx, outer_idx, apex_idx]
}

/// Index of the vertex in `vertices` closest to `target` in XZ.
///
/// Uses `f32::total_cmp` so NaN coordinates can't cause a panic; they
/// simply sort to one end like any other value.
fn nearest_vertex_xz(vertices: &[Vec3; 3], target: Vec3) -> usize {
    let t = Vec2::new(target.x, target.z);
    let d = [
        Vec2::new(vertices[0].x, vertices[0].z).distance_squared(t),
        Vec2::new(vertices[1].x, vertices[1].z).distance_squared(t),
        Vec2::new(vertices[2].x, vertices[2].z).distance_squared(t),
    ];
    let mut best = 0;
    if d[1].total_cmp(&d[best]).is_lt() {
        best = 1;
    }
    if d[2].total_cmp(&d[best]).is_lt() {
        best = 2;
    }
    best
}

/// Scalar `f32` math routed through either the platform `std`
/// implementations or `libm`.
///
/// The `std` methods are used on `std` builds unless the `libm` feature is
/// set to force bit-reproducible `libm` results everywhere; `no_std` builds
/// always use `libm`. Exactly one backend module is compiled per build.
///
/// glam's own vector math (`length`, `cross`, `normalize_or`, …) is routed
/// separately by glam's matching feature; this module only covers the scalar
/// operations the crate calls directly.
mod fmath {
    #[cfg(all(feature = "std", not(feature = "libm")))]
    mod imp {
        #[inline]
        pub fn sqrt(x: f32) -> f32 {
            x.sqrt()
        }
        #[inline]
        pub fn log2(x: f32) -> f32 {
            x.log2()
        }
        #[inline]
        pub fn mul_add(a: f32, b: f32, c: f32) -> f32 {
            a.mul_add(b, c)
        }
        #[inline]
        pub fn abs(x: f32) -> f32 {
            x.abs()
        }
        #[inline]
        pub fn round(x: f32) -> f32 {
            x.round()
        }
        #[inline]
        pub fn max(a: f32, b: f32) -> f32 {
            a.max(b)
        }
    }

    #[cfg(any(feature = "libm", not(feature = "std")))]
    mod imp {
        #[inline]
        pub fn sqrt(x: f32) -> f32 {
            libm::sqrtf(x)
        }
        #[inline]
        pub fn log2(x: f32) -> f32 {
            libm::log2f(x)
        }
        #[inline]
        pub fn mul_add(a: f32, b: f32, c: f32) -> f32 {
            libm::fmaf(a, b, c)
        }
        #[inline]
        pub fn abs(x: f32) -> f32 {
            libm::fabsf(x)
        }
        #[inline]
        pub fn round(x: f32) -> f32 {
            libm::roundf(x)
        }
        #[inline]
        pub fn max(a: f32, b: f32) -> f32 {
            libm::fmaxf(a, b)
        }
    }

    pub use imp::{abs, log2, max, mul_add, round, sqrt};

    /// Integer power by binary exponentiation.
    ///
    /// Implemented directly (rather than via `f32::powi` / `libm::powf`) so the
    /// result is identical on both backends — the only exponents the crate
    /// raises are small non-negative subdivision levels.
    #[inline]
    pub fn powi(mut base: f32, mut exp: i32) -> f32 {
        if exp < 0 {
            base = 1.0 / base;
            exp = exp.wrapping_neg();
        }
        let mut acc = 1.0f32;
        while exp > 0 {
            if exp & 1 == 1 {
                acc *= base;
            }
            base *= base;
            exp >>= 1;
        }
        acc
    }
}

fn cross_xz(a: Vec2, b: Vec2) -> f32 {
    fmath::mul_add(a.x, b.y, -(a.y * b.x))
}

/// Barycentric weights of `p` with respect to triangle `(a, b, c)` in the
/// XZ plane, as `(wa, wb, wc)` with `wa + wb + wc == 1`.
///
/// Returns `None` if the triangle is degenerate (zero area). The weights
/// are not clamped, so a point outside the triangle yields a weight
/// outside `[0, 1]`; callers that have already located the containing
/// leaf can interpolate directly.
fn barycentric_coords_xz(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> Option<(f32, f32, f32)> {
    let v0 = b - a;
    let v1 = c - a;
    let denom = cross_xz(v0, v1);
    if denom == 0.0 {
        return None;
    }
    let v2 = p - a;
    let wb = cross_xz(v2, v1) / denom;
    let wc = cross_xz(v0, v2) / denom;
    let wa = 1.0 - wb - wc;
    Some((wa, wb, wc))
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::SQRT_2;

    /// Unit square on the XZ plane: a=(0,0,0) b=(1,0,0) c=(1,0,1) d=(0,0,1).
    fn unit_square() -> [Vec3; 4] {
        [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 1.0),
        ]
    }

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    fn approx_vec2(a: Vec2, b: Vec2) -> bool {
        approx_eq(a.x, b.x) && approx_eq(a.y, b.y)
    }

    // ── NodeId ──────────────────────────────────────────────────────

    #[test]
    fn node_id_depth() {
        assert_eq!(NodeId::TRI_0.depth(), 1);
        assert_eq!(NodeId::TRI_1.depth(), 1);
        assert_eq!(NodeId::TRI_0.left().depth(), 2);
        assert_eq!(NodeId::TRI_1.right().depth(), 2);
        assert_eq!(NodeId::TRI_0.left().left().depth(), 3);
    }

    #[test]
    fn node_id_parent() {
        assert_eq!(NodeId::TRI_0.parent(), None);
        assert_eq!(NodeId::TRI_1.parent(), None);
        assert_eq!(NodeId::TRI_0.left().parent(), Some(NodeId::TRI_0));
        assert_eq!(NodeId::TRI_1.right().parent(), Some(NodeId::TRI_1));
        assert_eq!(
            NodeId::TRI_0.left().right().parent(),
            Some(NodeId::TRI_0.left())
        );
    }

    #[test]
    fn node_id_children_round_trip() {
        let id = NodeId::TRI_0;
        let [l, r] = id.children();
        assert_eq!(l.parent(), Some(id));
        assert_eq!(r.parent(), Some(id));
        assert!(l.is_left());
        assert!(!r.is_left());
    }

    #[test]
    fn node_id_root_triangle() {
        assert_eq!(NodeId::TRI_0.root_triangle(), 0);
        assert_eq!(NodeId::TRI_1.root_triangle(), 1);
        assert_eq!(NodeId::TRI_0.left().right().root_triangle(), 0);
        assert_eq!(NodeId::TRI_1.right().left().root_triangle(), 1);
    }

    #[test]
    fn node_id_path_bits() {
        // TRI_0 = 0b10 → path [0]
        let bits: Vec<_> = NodeId::TRI_0.path_bits().collect();
        assert_eq!(bits, vec![0]);
        // TRI_1 = 0b11 → path [1]
        let bits: Vec<_> = NodeId::TRI_1.path_bits().collect();
        assert_eq!(bits, vec![1]);
        // TRI_0.left().right() = 0b1001 → path [0,0,1]
        let bits: Vec<_> = NodeId::TRI_0.left().right().path_bits().collect();
        assert_eq!(bits, vec![0, 0, 1]);
    }

    #[test]
    fn subdivision_ids_root() {
        // A root triangle only contains itself.
        let ids = NodeId::TRI_0.subdivision_ids();
        assert_eq!(ids, vec![NodeId::TRI_0]);
    }

    #[test]
    fn subdivision_ids_depth2() {
        // Subdividing TRI_0 creates both children.
        let ids = NodeId::TRI_0.left().subdivision_ids();
        assert_eq!(ids.len(), 3);
        assert!(ids.contains(&NodeId::TRI_0));
        assert!(ids.contains(&NodeId::TRI_0.left()));
        assert!(ids.contains(&NodeId::TRI_0.right()));
    }

    #[test]
    fn subdivision_ids_depth4() {
        // TRI_0 → left → right → left  (0b10010, depth 4)
        let target = NodeId::TRI_0.left().right().left();
        let ids = target.subdivision_ids();
        // 1 root + 2 children per level × 3 levels = 7
        assert_eq!(ids.len(), 7);
        assert!(ids.contains(&NodeId::TRI_0));
        assert!(ids.contains(&NodeId::TRI_0.left()));
        assert!(ids.contains(&NodeId::TRI_0.right()));
        assert!(ids.contains(&NodeId::TRI_0.left().left()));
        assert!(ids.contains(&NodeId::TRI_0.left().right()));
        assert!(ids.contains(&NodeId::TRI_0.left().right().left()));
        assert!(ids.contains(&NodeId::TRI_0.left().right().right()));
        // The target itself is included.
        assert!(ids.contains(&target));
    }

    #[test]
    fn subdivision_leaf_ids_root() {
        let ids = NodeId::TRI_0.subdivision_leaf_ids();
        assert_eq!(ids, vec![NodeId::TRI_0]);
    }

    #[test]
    fn subdivision_leaf_ids_depth2() {
        let ids = NodeId::TRI_0.left().subdivision_leaf_ids();
        // Leaves: TRI_0.right() (off-path sibling) and TRI_0.left() (target).

        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0], NodeId::TRI_0.right());
        assert_eq!(ids[1], NodeId::TRI_0.left());
    }

    #[test]
    fn subdivision_leaf_ids_depth4() {
        let target = NodeId::TRI_0.left().right().left();
        let ids = target.subdivision_leaf_ids();
        // depth=4 → 3 off-path siblings + target = 4 leaves.
        assert_eq!(ids.len(), 4);
        assert_eq!(ids[0], NodeId::TRI_0.right());
        assert_eq!(ids[1], NodeId::TRI_0.left().left());
        assert_eq!(ids[2], NodeId::TRI_0.left().right().right());
        assert_eq!(ids[3], target);
    }

    #[test]
    fn subdivision_leaf_ids_subset_of_subdivision_ids() {
        let target = NodeId::TRI_1.right().left().right();
        let all = target.subdivision_ids();
        let leaves = target.subdivision_leaf_ids();
        for leaf in &leaves {
            assert!(all.contains(leaf));
        }
        // Every non-leaf in `all` must have both children in `all`.
        for id in &all {
            if !leaves.contains(id) {
                assert!(all.contains(&id.left()));
                assert!(all.contains(&id.right()));
            }
        }
    }

    // ── ChunkTriTree construction ───────────────────────────────────

    #[test]
    fn new_tree_has_two_leaves() {
        let tree = ChunkTriTree::<NodeId>::new(unit_square(), Diagonal::AC);
        assert_eq!(tree.len(), 2);
        assert!(tree.is_leaf(NodeId::TRI_0));
        assert!(tree.is_leaf(NodeId::TRI_1));
        assert_eq!(tree.leaves().count(), 2);
    }

    #[test]
    fn new_tree_shares_diagonal_corners() {
        // AC diagonal: TRI_0=[a,c,b], TRI_1=[a,d,c]; the `a` and `c`
        // corners are shared so the pool has 4 unique vertices total.
        let tree = ChunkTriTree::<NodeId>::new(unit_square(), Diagonal::AC);
        assert_eq!(tree.vertex_count(), 4);
    }

    #[test]
    fn new_tree_bd_diagonal() {
        let tree = ChunkTriTree::<NodeId>::new(unit_square(), Diagonal::BD);
        assert_eq!(tree.len(), 2);
        assert!(tree.is_leaf(NodeId::TRI_0));
        assert!(tree.is_leaf(NodeId::TRI_1));
        assert_eq!(tree.vertex_count(), 4);
    }

    // ── Size calculations ───────────────────────────────────────────

    #[test]
    fn chunk_side_unit_square() {
        let tree = ChunkTriTree::<NodeId>::new(unit_square(), Diagonal::AC);
        assert!(approx_eq(tree.chunk_side(), 1.0));
    }

    #[test]
    fn leg_and_base_at_level() {
        let tree = ChunkTriTree::<NodeId>::new(unit_square(), Diagonal::AC);
        // Level 1: leg = 1.0, base = √2
        assert!(approx_eq(tree.leg_at_level(1), 1.0));
        assert!(approx_eq(tree.base_at_level(1), SQRT_2));
        // Level 2: leg = 1/√2, base = 1
        assert!(approx_eq(tree.leg_at_level(2), 1.0 / SQRT_2));
        assert!(approx_eq(tree.base_at_level(2), 1.0));
        // Level 3: leg = 0.5, base = 1/√2
        assert!(approx_eq(tree.leg_at_level(3), 0.5));
        assert!(approx_eq(tree.base_at_level(3), 1.0 / SQRT_2));
    }

    #[test]
    fn level_for_square_edge_round_trip() {
        let tree = ChunkTriTree::<NodeId>::new(unit_square(), Diagonal::AC);
        for level in 1..=8 {
            let leg = tree.leg_at_level(level);
            assert_eq!(tree.level_for_square_edge(leg), level);
        }
    }

    #[test]
    fn level_for_base_round_trip() {
        let tree = ChunkTriTree::<NodeId>::new(unit_square(), Diagonal::AC);
        for level in 1..=8 {
            let base = tree.base_at_level(level);
            assert_eq!(tree.level_for_base(base), level);
        }
    }

    // ── vertices_of ─────────────────────────────────────────────────

    #[test]
    fn vertices_of_root_tris_ac() {
        let tree = ChunkTriTree::<NodeId>::new(unit_square(), Diagonal::AC);
        let [apex, bl, br] = tree.vertices_of(&NodeId::TRI_0);
        // AC tri 0: apex=b(1,0,0) base=a..c diagonal
        assert_eq!(apex, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(bl, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(br, Vec3::new(1.0, 0.0, 1.0));
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn vertices_of_children_cover_parent() {
        let tree = ChunkTriTree::<NodeId>::new(unit_square(), Diagonal::AC);
        let [apex, bl, br] = tree.vertices_of(&NodeId::TRI_0);
        let mid = (bl + br) * 0.5;

        let [c_apex_l, c_bl_l, c_br_l] = tree.vertices_of(&NodeId::TRI_0.left());
        let [c_apex_r, c_bl_r, c_br_r] = tree.vertices_of(&NodeId::TRI_0.right());

        // Both children share the midpoint as apex.
        assert!(c_apex_l.distance(mid) < 1e-5);
        assert!(c_apex_r.distance(mid) < 1e-5);

        // Left child keeps base_left, gains old apex as base_right.
        assert!(c_bl_l.distance(bl) < 1e-5);
        assert!(c_br_l.distance(apex) < 1e-5);

        // Right child gains old apex as base_left, keeps base_right.
        assert!(c_bl_r.distance(apex) < 1e-5);
        assert!(c_br_r.distance(br) < 1e-5);
    }

    // ── base_midpoint_xz / centroid_xz ──────────────────────────────

    #[test]
    fn base_midpoint_of_root_tri0() {
        let tree = ChunkTriTree::<NodeId>::new(unit_square(), Diagonal::AC);
        // Root tri 0 AC: base = a(0,0)..c(1,1) → midpoint (0.5, 0.5)
        let mid = tree.base_midpoint_xz(&NodeId::TRI_0);
        assert!(approx_vec2(mid, Vec2::new(0.5, 0.5)));
    }

    #[test]
    fn centroid_of_root_tri0() {
        let tree = ChunkTriTree::<NodeId>::new(unit_square(), Diagonal::AC);
        // Apex=b(1,0), bl=a(0,0), br=c(1,1) → centroid (2/3, 1/3)
        let cen = tree.centroid_xz(&NodeId::TRI_0);
        assert!(approx_vec2(cen, Vec2::new(2.0 / 3.0, 1.0 / 3.0)));
    }

    // ── id_at_xz ────────────────────────────────────────────────────

    #[test]
    fn id_at_xz_selects_correct_root_tri() {
        let tree = ChunkTriTree::<NodeId>::new(unit_square(), Diagonal::AC);
        // Point near b corner → tri 0
        let id = tree.id_at_xz(Vec2::new(0.9, 0.1), 1).unwrap();
        assert_eq!(id, NodeId::TRI_0);
        // Point near d corner → tri 1
        let id = tree.id_at_xz(Vec2::new(0.1, 0.9), 1).unwrap();
        assert_eq!(id, NodeId::TRI_1);
    }

    #[test]
    fn id_at_xz_deeper_levels() {
        let tree = ChunkTriTree::<NodeId>::new(unit_square(), Diagonal::AC);
        // Query at depth 2 — the centroid of the root tri 0 should stay
        // in the same sub-tree.
        let cen = tree.centroid_xz(&NodeId::TRI_0);
        let id = tree.id_at_xz(cen, 2).unwrap();
        assert_eq!(id.depth(), 2);
        assert_eq!(id.root_triangle(), 0);
    }

    #[test]
    fn id_at_xz_round_trip_with_centroid() {
        let tree = ChunkTriTree::<NodeId>::new(unit_square(), Diagonal::AC);
        for depth in 1..=6 {
            // Pick an arbitrary node and verify that its centroid maps back.
            let target = NodeId(0b10).left().right(); // depth 3 under tri 0
            if depth >= target.depth() {
                let cen = tree.centroid_xz(&target);
                let found = tree.id_at_xz(cen, target.depth()).unwrap();
                assert_eq!(found, target);
            }
        }
    }

    // ── leaf_at_xz ──────────────────────────────────────────────────

    #[test]
    fn leaf_at_xz_returns_root_leaf_when_unsplit() {
        let tree = ChunkTriTree::<NodeId>::new(unit_square(), Diagonal::AC);
        let id = tree.leaf_at_xz(Vec2::new(0.9, 0.1)).unwrap();
        assert_eq!(id, NodeId::TRI_0);
    }

    #[test]
    fn leaf_at_xz_follows_splits() {
        let mut tree = ChunkTriTree::<NodeId>::new(unit_square(), Diagonal::AC);
        tree.split(&NodeId::TRI_0).unwrap();

        // Centroid of the left child should resolve to the left child.
        let left_cen = tree.centroid_xz(&NodeId::TRI_0.left());
        let found = tree.leaf_at_xz(left_cen).unwrap();
        assert_eq!(found, NodeId::TRI_0.left());

        // Centroid of the right child should resolve to the right child.
        let right_cen = tree.centroid_xz(&NodeId::TRI_0.right());
        let found = tree.leaf_at_xz(right_cen).unwrap();
        assert_eq!(found, NodeId::TRI_0.right());

        // TRI_1 is still a leaf.
        let tri1_cen = tree.centroid_xz(&NodeId::TRI_1);
        let found = tree.leaf_at_xz(tri1_cen).unwrap();
        assert_eq!(found, NodeId::TRI_1);
    }

    // ── Diagonal::BD ────────────────────────────────────────────────

    #[test]
    fn bd_diagonal_id_at_xz() {
        let tree = ChunkTriTree::<NodeId>::new(unit_square(), Diagonal::BD);
        // Point near a corner (0,0) → tri 0 (a,b,d)
        let id = tree.id_at_xz(Vec2::new(0.1, 0.1), 1).unwrap();
        assert_eq!(id, NodeId::TRI_0);
        // Point near c corner (1,1) → tri 1 (b,c,d)
        let id = tree.id_at_xz(Vec2::new(0.9, 0.9), 1).unwrap();
        assert_eq!(id, NodeId::TRI_1);
    }

    #[test]
    fn id_at_xz_outside_returns_none() {
        let tree = ChunkTriTree::<NodeId>::new(unit_square(), Diagonal::AC);
        assert_eq!(tree.id_at_xz(Vec2::new(-0.5, 0.5), 1), None);
        assert_eq!(tree.id_at_xz(Vec2::new(1.5, 0.5), 1), None);
        assert_eq!(tree.id_at_xz(Vec2::new(0.5, -0.5), 1), None);
        assert_eq!(tree.id_at_xz(Vec2::new(0.5, 1.5), 1), None);
        assert_eq!(tree.id_at_xz(Vec2::new(2.0, 2.0), 3), None);

        let tree_bd = ChunkTriTree::<NodeId>::new(unit_square(), Diagonal::BD);
        assert_eq!(tree_bd.id_at_xz(Vec2::new(-0.1, 0.5), 1), None);
        assert_eq!(tree_bd.id_at_xz(Vec2::new(0.5, 1.1), 1), None);
    }

    #[test]
    fn leaf_at_xz_outside_returns_none() {
        let tree = ChunkTriTree::<NodeId>::new(unit_square(), Diagonal::AC);
        assert_eq!(tree.leaf_at_xz(Vec2::new(-0.5, 0.5)), None);
        assert_eq!(tree.leaf_at_xz(Vec2::new(1.5, 0.5)), None);
    }

    #[test]
    fn path_from_leaf_to_depth_starts_with_leaf() {
        let tree = ChunkTriTree::<NodeId>::new(unit_square(), Diagonal::AC);
        let leaf = NodeId::TRI_0.left().right();
        let xz = tree.centroid_xz(&leaf);

        let path = tree.path_from_leaf_to_depth(xz, 3).unwrap();

        assert_eq!(path, vec![NodeId::TRI_0, NodeId::TRI_0.left(), leaf]);
    }

    #[test]
    fn sparse_tree_height_queries_work_once_path_is_materialized() {
        let mut tree = ChunkTriTree::<NodeId>::new(unit_square(), Diagonal::AC);
        let leaf = NodeId::TRI_0.left().right();

        let mut path = tree
            .path_from_leaf_to_depth(tree.centroid_xz(&leaf), leaf.depth())
            .unwrap();
        // The path includes the target itself; pop it so we only
        // split its ancestors.
        path.pop();
        for id in path {
            tree.split(&id);
        }

        // Bump the y of every vertex of the materialised leaf via the
        // shared pool; the leaf's own [VertexId; 3] and the parent
        // path's surviving vertices all see the changes.
        let node = *tree.get(&leaf).unwrap();
        for (i, vid) in node.vertices.iter().enumerate() {
            tree.vertex_mut(*vid).position.y = [3.0, 5.0, 7.0][i];
        }

        let xz = tree.centroid_xz(&leaf);
        assert!(tree.height_at_xz(xz).is_some());
        assert_eq!(tree.leaf_at_xz(xz), Some(leaf));
    }

    // ── Sharing semantics ───────────────────────────────────────────

    #[test]
    fn split_shares_corner_vertices_with_parent() {
        let mut tree = ChunkTriTree::<NodeId>::new(unit_square(), Diagonal::AC);
        let parent = *tree.get(&NodeId::TRI_0).unwrap();
        let parent_ids: HashSet<VertexId> = parent.vertices.iter().copied().collect();

        tree.split(&NodeId::TRI_0).unwrap();

        let left = *tree.get(&NodeId::TRI_0.left()).unwrap();
        let right = *tree.get(&NodeId::TRI_0.right()).unwrap();
        let child_ids: HashSet<VertexId> = left
            .vertices
            .iter()
            .chain(right.vertices.iter())
            .copied()
            .collect();

        // Every original parent corner survives unchanged in the
        // child set (apex appears in both children, bl in left, br in
        // right).
        for pid in &parent_ids {
            assert!(child_ids.contains(pid), "parent vertex lost on split");
        }

        // The split should have created exactly one new vertex (the
        // base midpoint) — chunk has 4 corners + 1 midpoint = 5.
        assert_eq!(tree.vertex_count(), 5);
    }

    #[test]
    fn force_set_leaf_overwrite_with_shared_position_keeps_pool_consistent() {
        // Regression: when `force_set_leaf` is called twice on the
        // same id and the two calls share at least one position,
        // `set_leaf` used to release the prev leaf's vertices before
        // retaining the new ones. The shared slot would dip to rc=0,
        // be pushed onto `free_slots`, then be retained back to rc=1
        // — leaving it live AND in the free list. The next
        // `intern_vertex` would pop it and silently clobber an in-use
        // vertex, producing a leaf that references a slot whose
        // refcount is 0 (`vertices()` filters it out, and any caller
        // that builds a vertex-id remap from `vertices()` would fail
        // its lookup).
        let mut tree = ChunkTriTree::<NodeId>::new(unit_square(), Diagonal::AC);

        // Re-import TRI_0 with a triangle that shares corner `b` with
        // the root-init leaf. With Diagonal::AC, TRI_0 = [a, b, c] and
        // TRI_1 = [a, c, d], so `b` is referenced only by TRI_0 — its
        // slot drops to rc=0 on release, exposing the bug. (Sharing
        // `a` or `c` wouldn't trigger it, because TRI_1 still holds a
        // reference and rc never reaches 0.)
        let b = Vec3::new(1.0, 0.0, 0.0);
        let q = Vec3::new(0.5, 0.0, 0.0);
        let r = Vec3::new(0.7, 0.0, 0.2);
        tree.force_set_leaf(NodeId::TRI_0, [b, q, r], [Vec2::ZERO; 3], [Vec3::Y; 3]);

        // Drain `free_slots` past the poisoned slot. Demoting TRI_1
        // releases its 3 corners (pushing 3 more slots onto the LIFO
        // free list, on top of the poisoned slot from above), then
        // four `intern_vertex` calls with brand-new positions pop them
        // back out one by one. The fourth pop hits the poisoned slot,
        // which under the buggy ordering has refcount=1 and is still
        // referenced by TRI_0's leaf — overwriting it corrupts TRI_0.
        tree.force_set_interior(NodeId::TRI_1);
        let deep = NodeId::TRI_1.left().left();
        let mut cur = deep;
        while let Some(parent) = cur.parent() {
            if parent != NodeId::TRI_1 {
                tree.force_set_interior(parent);
            }
            cur = parent;
        }
        let p1 = Vec3::new(0.90, 0.5, 0.50);
        let p2 = Vec3::new(0.95, 0.5, 0.55);
        let p3 = Vec3::new(0.92, 0.5, 0.60);
        tree.force_set_leaf(deep, [p1, p2, p3], [Vec2::ZERO; 3], [Vec3::Y; 3]);
        // One more brand-new position to drain the fourth (poisoned) slot.
        let other_deep = NodeId::TRI_1.right().right();
        let mut cur = other_deep;
        while let Some(parent) = cur.parent() {
            if parent != NodeId::TRI_1 {
                tree.force_set_interior(parent);
            }
            cur = parent;
        }
        let p4 = Vec3::new(0.80, 0.5, 0.70);
        let p5 = Vec3::new(0.82, 0.5, 0.72);
        let p6 = Vec3::new(0.84, 0.5, 0.74);
        tree.force_set_leaf(other_deep, [p4, p5, p6], [Vec2::ZERO; 3], [Vec3::Y; 3]);

        // Every leaf vertex must still resolve to a live slot, and
        // TRI_0's positions must be unchanged.
        let live: HashSet<VertexId> = tree.vertices().map(|(id, _)| id).collect();
        for (nid, node) in tree.leaves() {
            for v in node.vertices {
                assert!(live.contains(&v), "leaf {nid:?} references dead slot {v:?}");
            }
        }
        assert_eq!(tree.leaf_positions(NodeId::TRI_0).unwrap(), [b, q, r]);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn shared_corner_edit_propagates_to_all_leaves() {
        let mut tree = ChunkTriTree::<NodeId>::new(unit_square(), Diagonal::AC);
        tree.split(&NodeId::TRI_0).unwrap();

        // Find the shared apex (parent's old apex, now the base_right
        // of left child and base_left of right child). Both children
        // also share the new base midpoint, so filter to the corner
        // at the parent's old apex position `b = (1, 0, 0)`.
        let left = *tree.get(&NodeId::TRI_0.left()).unwrap();
        let right = *tree.get(&NodeId::TRI_0.right()).unwrap();
        let mut shared = None;
        for v in left.vertices {
            if right.vertices.contains(&v) {
                let p = tree.vertex(v).position;
                if approx_eq(p.x, 1.0) && approx_eq(p.z, 0.0) {
                    shared = Some(v);
                    break;
                }
            }
        }
        let shared = shared.expect("children must share the apex");

        tree.vertex_mut(shared).position.y = 4.2;

        // Both children now read the same updated y on their shared corner.
        let left_y = tree
            .leaf_positions(NodeId::TRI_0.left())
            .unwrap()
            .iter()
            .find(|p| p.x == 1.0 && p.z == 0.0)
            .unwrap()
            .y;
        let right_y = tree
            .leaf_positions(NodeId::TRI_0.right())
            .unwrap()
            .iter()
            .find(|p| p.x == 1.0 && p.z == 0.0)
            .unwrap()
            .y;
        assert!(approx_eq(left_y, 4.2));
        assert!(approx_eq(right_y, 4.2));
    }

    // ── Merging ─────────────────────────────────────────────────────

    #[test]
    fn merge_flat_fails_when_not_split() {
        let mut tree = ChunkTriTree::<NodeId>::new(unit_square(), Diagonal::AC);
        assert!(!tree.merge_flat(NodeId::TRI_0, 0.01));
    }

    #[test]
    fn merge_flat_undoes_a_split_on_flat_ground() {
        let mut tree = ChunkTriTree::<NodeId>::new(unit_square(), Diagonal::AC);
        assert!(tree.split(&NodeId::TRI_0).is_some());
        assert_eq!(tree.leaves().count(), 3);
        assert!(tree.merge_flat(NodeId::TRI_0, 1e-4));
        assert!(tree.is_leaf(NodeId::TRI_0));
        assert_eq!(tree.leaves().count(), 2);
        // Midpoint slot should have been freed.
        assert_eq!(tree.vertex_count(), 4);
    }

    #[test]
    fn merge_flat_preserves_per_vertex_normals_on_corners() {
        let mut tree = ChunkTriTree::<NodeId>::new(unit_square(), Diagonal::AC);

        // Stamp distinct normals onto TRI_0's three corner slots so we
        // can detect whether they survive the split-merge round trip.
        let corner_ids = tree.get(&NodeId::TRI_0).unwrap().vertices;
        let normals = [
            Vec3::new(0.1, 1.0, 0.0).normalize(),
            Vec3::new(-0.1, 1.0, 0.2).normalize(),
            Vec3::new(0.0, 1.0, -0.3).normalize(),
        ];
        for (i, vid) in corner_ids.iter().enumerate() {
            tree.vertex_mut(*vid).normal = normals[i];
        }
        let expected: HashMap<XzKey, Vec3> = corner_ids
            .iter()
            .map(|vid| {
                let v = tree.vertex(*vid);
                (xz_key(v.position), v.normal)
            })
            .collect();

        assert!(tree.split(&NodeId::TRI_0).is_some());
        assert!(tree.merge_flat(NodeId::TRI_0, 1e-4));

        let merged_ids = tree.get(&NodeId::TRI_0).unwrap().vertices;
        for vid in merged_ids {
            let v = tree.vertex(vid);
            let want = expected[&xz_key(v.position)];
            assert!(
                (v.normal - want).length() < 1e-4,
                "normal {:?} mismatched at {:?}, expected {want:?}",
                v.normal,
                v.position
            );
        }
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn merge_flat_refuses_when_midpoint_is_raised() {
        let mut tree = ChunkTriTree::<NodeId>::new(unit_square(), Diagonal::AC);
        tree.split(&NodeId::TRI_0);

        // Find the midpoint vertex and raise it. Because the midpoint
        // is shared between the two children, raising it once is
        // enough — that's exactly the property we want to test.
        let [_, geo_bl, geo_br] = tree.vertices_of(&NodeId::TRI_0);
        let geo_mid = (geo_bl + geo_br) * 0.5;
        let mid_key = xz_key(geo_mid);
        let mid_id = *tree.by_xz.get(&mid_key).unwrap();
        tree.vertex_mut(mid_id).position.y = 1.0;

        assert!(!tree.merge_flat(NodeId::TRI_0, 0.1));
        assert!(tree.has_children(NodeId::TRI_0));
    }

    #[test]
    fn simplify_flat_collapses_full_subdivision() {
        let mut tree = ChunkTriTree::<NodeId>::new(unit_square(), Diagonal::AC);
        for id in NodeId::TRI_0.subdivision_ids() {
            tree.split(&id);
        }
        assert!(tree.leaves().count() > 2);
        let merged = tree.simplify_flat(1e-4);
        assert!(merged > 0);
        assert_eq!(tree.leaves().count(), 2);
        assert!(tree.is_leaf(NodeId::TRI_0));
    }

    #[test]
    fn simplify_flat_preserves_nonflat_subtrees() {
        let mut tree = ChunkTriTree::<NodeId>::new(unit_square(), Diagonal::AC);
        tree.split(&NodeId::TRI_0);
        tree.split(&NodeId::TRI_0.left());

        // Raise the midpoint between TRI_0.left's children — that inner
        // split must survive, so TRI_0 itself cannot collapse.
        let [_, bl, br] = tree.vertices_of(&NodeId::TRI_0.left());
        let mid = (bl + br) * 0.5;
        let mid_id = *tree.by_xz.get(&xz_key(mid)).unwrap();
        tree.vertex_mut(mid_id).position.y = 2.0;

        tree.simplify_flat(0.1);
        assert!(tree.has_children(NodeId::TRI_0));
        assert!(tree.has_children(NodeId::TRI_0.left()));
    }
}
