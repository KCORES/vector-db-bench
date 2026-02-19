use crate::api::*;
use rayon::prelude::*;
use parking_lot::{RwLock, RwLockUpgradableReadGuard};
use crate::distance::l2_distance;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use rand::seq::SliceRandom;
use rand::thread_rng;

const NUM_CLUSTERS: usize = 2048; // Increased sqrt(1M) approx for finer granularity
const N_PROBE: usize = 48; // Decrease probe count to boost speed, while fine granularity keeps recall?

pub struct VectorDB {
    data: RwLock<Vec<(u64, Vec<f32>)>>,
    index: RwLock<IVFIndex>,
}

#[derive(Default)]
struct IVFIndex {
    built: bool,
    // [NUM_CLUSTERS][128], aligned? 
    // Storing as flat vector might help cache prefetching if we access them sequentially?
    // But we random access them.
    centroids: Vec<Vec<f32>>, 
    lists: Vec<InvertedList>,
}

#[derive(Default)]
struct InvertedList {
    ids: Vec<u64>,
    vectors: Vec<f32>, // flattened
}

#[derive(PartialEq)]
struct OrderedFloat(f64);

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

#[derive(Eq, PartialEq)]
struct HeapItem {
    distance: OrderedFloat,
    id: u64,
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.cmp(&other.distance)
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl VectorDB {
    pub fn new() -> Self {
        VectorDB {
            data: RwLock::new(Vec::new()),
            index: RwLock::new(IVFIndex::default()),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        if vector.len() != 128 { return; }
        self.data.write().push((id, vector));
    }

    pub fn bulk_insert(&self, new_vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let count = new_vectors.len();
        let mut data = self.data.write();
        for (id, vec) in new_vectors {
            if vec.len() == 128 {
                data.push((id, vec));
            }
        }
        count
    }

    fn train_and_build(&self) {
        let upgradeable_index = self.index.upgradable_read();
        
        if upgradeable_index.built {
            return;
        }

        let mut data_lock = self.data.write();
        if data_lock.is_empty() {
            return;
        }
        
        let mut index_lock = RwLockUpgradableReadGuard::upgrade(upgradeable_index);
        let total_vecs = data_lock.len();

        let num_clusters = NUM_CLUSTERS;
        let mut rng = thread_rng();
        
        // Random centroids init
        let mut centroids: Vec<Vec<f32>> = (0..num_clusters)
            .map(|_| {
                let idx = rng.gen_range(0..total_vecs);
                data_lock[idx].1.clone()
            })
            .collect();
            
        // No K-Means refinement for speed. Rely on N_PROBE.
        
        index_lock.centroids = centroids;
        index_lock.lists = (0..num_clusters).map(|_| InvertedList::default()).collect();

        // Parallel assignment
        // Chunk the data to reduce overhead of thread spawning if too small
        // But rayon handles work stealing fine.
        let assignments: Vec<usize> = data_lock.par_iter().map(|(_, vec)| {
            let mut best_dist = f64::MAX;
            let mut best_c = 0;
            // Unroll loop manually or rely on compiler?
            // This inner loop is small (2048 iters).
            for (c_idx, c_vec) in index_lock.centroids.iter().enumerate() {
                let d = l2_distance(vec, c_vec);
                if d < best_dist {
                    best_dist = d;
                    best_c = c_idx;
                }
            }
            best_c
        }).collect();

        for (i, cluster_idx) in assignments.into_iter().enumerate() {
            let (id, vec) = &data_lock[i];
            index_lock.lists[cluster_idx].ids.push(*id);
            index_lock.lists[cluster_idx].vectors.extend_from_slice(vec);
        }

        index_lock.built = true;
        data_lock.clear();
    }

    pub fn search(&self, query_vector: &[f32], top_k: u32) -> Vec<SearchResult> {
        // Trigger check
        if self.data.read().len() > 10_000 {
             self.train_and_build();
        }

        let index = self.index.read();
        let staging = self.data.read();

        if !index.built {
            if staging.is_empty() { return Vec::new(); }
            let mut results: Vec<(u64, f64)> = staging.par_iter()
                .map(|(id, vec)| (*id, l2_distance(query_vector, vec)))
                .collect();
            results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            return results.into_iter().take(top_k as usize).map(|(id, d)| SearchResult{id, distance: d}).collect();
        }

        // 1. Find nearest centroids
        // Compute distances to all centroids
        // Flattened centroids might be faster?
        let mut centroid_dists: Vec<(usize, f64)> = index.centroids.iter()
            .enumerate()
            .map(|(i, c)| (i, l2_distance(query_vector, c)))
            .collect();
        
        // We only need top N_PROBE
        centroid_dists.select_nth_unstable_by(N_PROBE, |a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        let probes = std::cmp::min(N_PROBE, centroid_dists.len());

        let candidates_indices: Vec<usize> = centroid_dists.iter().take(probes).map(|(i, _)| *i).collect();

        let k = top_k as usize;
        
        // Use Rayon for parallel search of the lists
        // We create a thread-local max-heap
        
        let top_k_heap = candidates_indices.into_par_iter()
            .fold(
                || BinaryHeap::with_capacity(k + 1),
                |mut heap: BinaryHeap<HeapItem>, idx| {
                    let list = &index.lists[idx];
                    let vectors = &list.vectors;
                    let ids = &list.ids;
                    let n = ids.len();
                    
                    // We iterate sequentially over chunks in list
                    // vectors is flattened.
                    for i in 0..n {
                        let vec_slice = &vectors[i*128..(i+1)*128];
                        let dist = l2_distance(query_vector, vec_slice);
                        
                        if heap.len() < k {
                            heap.push(HeapItem { distance: OrderedFloat(dist), id: ids[i] });
                        } else if let Some(max_item) = heap.peek() {
                            if dist < max_item.distance.0 {
                                heap.pop();
                                heap.push(HeapItem { distance: OrderedFloat(dist), id: ids[i] });
                            }
                        }
                    }
                    heap
                }
            )
            .reduce(
                 || BinaryHeap::with_capacity(k),
                 |mut heap1, heap2| {
                    for item in heap2 {
                        if heap1.len() < k {
                            heap1.push(item);
                        } else if let Some(max_item) = heap1.peek() {
                             if item.distance < max_item.distance {
                                heap1.pop();
                                heap1.push(item);
                            }
                        }
                    }
                    heap1
                 }
            );

        // Merge with staging results
        // Staging is usually empty or small after bulk load
        let mut final_heap = top_k_heap;
        if !staging.is_empty() {
             for (id, vec) in staging.iter() {
                  let dist = l2_distance(query_vector, vec);
                   if final_heap.len() < k {
                        final_heap.push(HeapItem { distance: OrderedFloat(dist), id: *id });
                    } else if let Some(max_item) = final_heap.peek() {
                        if dist < max_item.distance.0 {
                            final_heap.pop();
                            final_heap.push(HeapItem { distance: OrderedFloat(dist), id: *id });
                        }
                    }
             }
        }

        // Output sorted
        final_heap.into_sorted_vec().into_iter()
            .map(|item| SearchResult { id: item.id, distance: item.distance.0 })
            .collect()
    }
}

use rand::Rng; // Add Rng trait usage
