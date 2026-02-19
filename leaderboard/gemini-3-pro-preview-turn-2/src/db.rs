use crate::api::*;
use rayon::prelude::*;
use parking_lot::RwLock;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const DIM: usize = 128;

pub struct VectorDB {
    ids: RwLock<Vec<u64>>,
    vectors: RwLock<Vec<f32>>,
}

#[derive(PartialEq, Clone, Copy)]
struct OrderedSearchResult {
    id: u64,
    distance: f32, 
}

impl Eq for OrderedSearchResult {}

// MaxHeap behavior: we want popping to remove the LARGEST distance.
// So Ord should be normal float comparison.
// When we keep Top K smallest, we check if new_dist < heap.peek() (max).
impl PartialOrd for OrderedSearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for OrderedSearchResult {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.partial_cmp(&other.distance).unwrap_or(Ordering::Equal)
    }
}

impl VectorDB {
    pub fn new() -> Self {
        VectorDB {
            ids: RwLock::new(Vec::new()),
            vectors: RwLock::new(Vec::new()),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        if vector.len() != DIM {
            panic!("Vector dimension mismatch");
        }
        let mut ids = self.ids.write();
        let mut vectors = self.vectors.write();
        ids.push(id);
        vectors.extend(vector);
    }

    pub fn bulk_insert(&self, new_vectors: Vec<(u64, Vec<f32>)>) -> usize {
        if new_vectors.is_empty() { return 0; }
        let count = new_vectors.len();
        let mut ids = self.ids.write();
        let mut vectors = self.vectors.write();
        
        ids.reserve(count);
        vectors.reserve(count * DIM);

        for (id, vec) in new_vectors {
            if vec.len() != DIM { panic!("Vector dimension mismatch"); }
            ids.push(id);
            vectors.extend(vec);
        }
        count
    }

    pub fn search(&self, query: &[f32], top_k: u32) -> Vec<SearchResult> {
        if top_k == 0 { return vec![]; }
        if query.len() != DIM { return vec![]; }
        
        let ids_guard = self.ids.read();
        let vectors_guard = self.vectors.read();
        
        let ids = &*ids_guard;
        let vectors = &*vectors_guard;
        
        // Ensure we have data
        if ids.is_empty() { return vec![]; }

        let num_vectors = ids.len();
        
        // Parallel iteration
        // Chunk size tuning: 
        // 1M vectors. If we have 4 threads, 250k per thread.
        // We can let Rayon decide or enforce chunks.
        // par_chunks() on a slice usually works well.
        // But we need to iterate `vectors` and `ids` in sync.
        // `ids` is Vec<u64>, `vectors` is Vec<f32> (flat).
        
        // To avoid excessive id lookups, we scan vectors and only lookup id for candidates.
        
        // How many vectors per rayon task?
        // L2 cache is 1MB / core. 256KB vectors = 2048 vectors.
        // 36MB L3 shared. 
        // Let's try explicit chunking to ensure we have enough work per task but not too much loop overhead.
        // 4096 vectors * 512 bytes = 2MB. 
        
        let chunk_size = 4096; 
        
        let top_k_heap = vectors.par_chunks(chunk_size * DIM)
            .zip(ids.par_chunks(chunk_size))
            .map(|(vec_chunk, id_chunk)| {
                let mut heap = BinaryHeap::with_capacity((top_k + 1) as usize);
                
                // Use unsafe AVX implementation
                unsafe {
                    search_chunk_avx512(query, vec_chunk, id_chunk, &mut heap, top_k as usize);
                }
                
                heap
            })
            .reduce(
                || BinaryHeap::with_capacity((top_k + 1) as usize),
                |mut heap1, heap2| {
                    for item in heap2 {
                        heap1.push(item);
                        if heap1.len() > top_k as usize {
                            heap1.pop();
                        }
                    }
                    heap1
                }
            );

        let mut sorted_results = top_k_heap.into_sorted_vec();
        
        sorted_results.iter().map(|r| SearchResult { 
            id: r.id, 
            distance: (r.distance as f64).sqrt() 
        }).collect()
    }
}

// AVX-512 optimized search for a chunk
#[target_feature(enable = "avx512f,avx512dq")]
unsafe fn search_chunk_avx512(
    query: &[f32], 
    vectors: &[f32], 
    ids: &[u64], 
    heap: &mut BinaryHeap<OrderedSearchResult>, 
    k: usize
) {
    // 1. Load query into registers (8 ZMM registers for 128 floats)
    // We assume query is 128 floats.
    let q_ptr = query.as_ptr();
    let q0 = _mm512_loadu_ps(q_ptr);
    let q1 = _mm512_loadu_ps(q_ptr.add(16));
    let q2 = _mm512_loadu_ps(q_ptr.add(32));
    let q3 = _mm512_loadu_ps(q_ptr.add(48));
    let q4 = _mm512_loadu_ps(q_ptr.add(64));
    let q5 = _mm512_loadu_ps(q_ptr.add(80));
    let q6 = _mm512_loadu_ps(q_ptr.add(96));
    let q7 = _mm512_loadu_ps(q_ptr.add(112));

    let n = ids.len();
    let mut v_ptr = vectors.as_ptr();
    
    // Maintain a local max distance to avoid heap operations
    let mut max_dist = f32::MAX;
    if heap.len() == k {
        if let Some(top) = heap.peek() {
            max_dist = top.distance;
        }
    }

    for i in 0..n {
        // Calculate L2^2 distance
        // Unroll 8 ZMM loads
        let v0 = _mm512_loadu_ps(v_ptr);
        let v1 = _mm512_loadu_ps(v_ptr.add(16));
        let v2 = _mm512_loadu_ps(v_ptr.add(32));
        let v3 = _mm512_loadu_ps(v_ptr.add(48));
        let v4 = _mm512_loadu_ps(v_ptr.add(64));
        let v5 = _mm512_loadu_ps(v_ptr.add(80));
        let v6 = _mm512_loadu_ps(v_ptr.add(96));
        let v7 = _mm512_loadu_ps(v_ptr.add(112));
        
        let d0 = _mm512_sub_ps(q0, v0);
        let d1 = _mm512_sub_ps(q1, v1);
        let d2 = _mm512_sub_ps(q2, v2);
        let d3 = _mm512_sub_ps(q3, v3);
        let d4 = _mm512_sub_ps(q4, v4);
        let d5 = _mm512_sub_ps(q5, v5);
        let d6 = _mm512_sub_ps(q6, v6);
        let d7 = _mm512_sub_ps(q7, v7);
        
        // Fused Multiply-Add: sum = diff * diff + 0.0
        // We can accumulate into a single register partially
        // But it's better to accumulate parallelly then reduce at end?
        // Or just let CPU pipelining handle it.
        // _mm512_fmadd_ps(a, b, c) -> a*b + c
        
        let s0 = _mm512_mul_ps(d0, d0);
        let s1 = _mm512_mul_ps(d1, d1);
        let s2 = _mm512_mul_ps(d2, d2);
        let s3 = _mm512_mul_ps(d3, d3);
        let s4 = _mm512_mul_ps(d4, d4);
        let s5 = _mm512_mul_ps(d5, d5);
        let s6 = _mm512_mul_ps(d6, d6);
        let s7 = _mm512_mul_ps(d7, d7);
        
        // Reduction tree
        let sum0 = _mm512_add_ps(s0, s1);
        let sum1 = _mm512_add_ps(s2, s3);
        let sum2 = _mm512_add_ps(s4, s5);
        let sum3 = _mm512_add_ps(s6, s7);
        
        let sum01 = _mm512_add_ps(sum0, sum1);
        let sum23 = _mm512_add_ps(sum2, sum3);
        
        let sum_total = _mm512_add_ps(sum01, sum23);
        
        let dist = _mm512_reduce_add_ps(sum_total);

        // Heap logic
        if dist < max_dist {
            let id = *ids.get_unchecked(i); // unsafe get for speed
            heap.push(OrderedSearchResult { id, distance: dist });
            if heap.len() > k {
                heap.pop();
                if let Some(top) = heap.peek() {
                    max_dist = top.distance;
                }
            } else if heap.len() == k {
                 if let Some(top) = heap.peek() {
                    max_dist = top.distance;
                }
            }
        }

        v_ptr = v_ptr.add(128);
    }
}
