// Tuning parameters for the final push
// Good progress. QPS ~1500, Recall 95.4%.
// Profile shows 80% time in AVX-512 distance calculation. This is ideal.
// The overhead of framework/sorting is minimal.
// To get higher QPS, we must reduce the number of distance calculations.
// Current config: Clusters=4096, Probe=64.
// Vectors per cluster = 244.
// Scanned vectors = 64 * 244 = 15616.
// Centroid scan = 4096.
// Total = 19712 per query.
// Can we reduce probes?
// If we reduce probes to 50: 50 * 244 = 12200. Total 16296. (17% less work).
// Recall 95.4% is close to boundary. 
// Maybe we can improve centroid assignment quality so we need fewer probes?
// But assignment is random right now.
// Doing a single K-Means iteration during build might improve cluster quality significantly, allowing fewer probes.
// "Train" is called once. Adding a little cost there is fine if it speeds up search.
// Let's implement 2 iterations of K-Means.

const NUM_CLUSTERS: usize = 4096;
const N_PROBE: usize = 50; // Try reducing probe count with better centroids

use crate::api::*;
use rayon::prelude::*;
use parking_lot::{RwLock, RwLockUpgradableReadGuard};
use crate::distance::l2_distance;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;

pub struct VectorDB {
    data: RwLock<Vec<(u64, Vec<f32>)>>,
    index: RwLock<IVFIndex>,
}

#[derive(Default)]
struct IVFIndex {
    built: bool,
    centroids: Vec<Vec<f32>>, 
    lists: Vec<InvertedList>,
}

#[derive(Default)]
struct InvertedList {
    ids: Vec<u64>,
    vectors: Vec<f32>, 
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
        
        // Initial random centroids from data
        let mut centroids: Vec<Vec<f32>> = (0..num_clusters)
            .map(|_| {
                let idx = rng.gen_range(0..total_vecs);
                data_lock[idx].1.clone()
            })
            .collect();

        // Single Pass K-Means Refinement
        // We use a subset of data to refine centroids to keep it fast
        let sample_size = std::cmp::min(total_vecs, 50_000); 
        // Just pick first sample_size for speed, shuffling is expensive and data is likely random enough or order doesn't matter much
        // Actually, let's step through data to be robust to sorted input
        let step = total_vecs / sample_size;
        
        // Assign samples to nearest centroid and accumulate
        // Parallel reduce is better here
        
        // (Cluster Index, (Count, SumVector))
        // This is a bit heavy to implement manually with Rayon fold/reduce.
        // Let's do a simple shared accumulator with atomics? No, floats. 
        // Mutex on each centroid? Too much contention.
        // Thread-local accumulation then merge.
        
        let new_centroids_sum: Vec<Vec<f32>> = (0..num_clusters).map(|_| vec![0.0; 128]).collect();
        let new_centroids_count: Vec<usize> = vec![0; num_clusters];
        
        let (final_sums, final_counts) = (0..sample_size).into_par_iter()
            .fold(
                || (vec![vec![0.0; 128]; num_clusters], vec![0usize; num_clusters]),
                |(mut sums, mut counts), i| {
                    let real_idx = (i * step) % total_vecs;
                    let vec = &data_lock[real_idx].1;
                    
                    let mut best_dist = f64::MAX;
                    let mut best_c = 0;
                    for (c_idx, c_vec) in centroids.iter().enumerate() {
                        let d = l2_distance(vec, c_vec);
                        if d < best_dist {
                            best_dist = d;
                            best_c = c_idx;
                        }
                    }
                    
                    // Accumulate
                    for k in 0..128 {
                        sums[best_c][k] += vec[k];
                    }
                    counts[best_c] += 1;
                    (sums, counts)
                }
            )
            .reduce(
                || (vec![vec![0.0; 128]; num_clusters], vec![0usize; num_clusters]),
                |(mut s1, mut c1), (s2, c2)| {
                   for i in 0..num_clusters {
                       c1[i] += c2[i];
                       for k in 0..128 {
                           s1[i][k] += s2[i][k];
                       }
                   }
                   (s1, c1)
                }
            );
            
         // Update centroids
         for i in 0..num_clusters {
             if final_counts[i] > 0 {
                 let inv = 1.0 / final_counts[i] as f32;
                 for k in 0..128 {
                     centroids[i][k] = final_sums[i][k] * inv;
                 }
             }
         }
            
        index_lock.centroids = centroids;
        index_lock.lists = (0..num_clusters).map(|_| InvertedList::default()).collect();

        // Assign all data
        let assignments: Vec<usize> = data_lock.par_iter().map(|(_, vec)| {
            let mut best_dist = f64::MAX;
            let mut best_c = 0;
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
        let mut data_len = 0;
        {
             let data = self.data.read();
             data_len = data.len();
        }
        if data_len > 10_000 {
            self.train_and_build();
        }

        let index = self.index.read();
        let staging = self.data.read();

        if !index.built {
            if staging.is_empty() { return Vec::new(); }
            let mut results: Vec<(u64, f64)> = staging.par_iter()
                .map(|(id, vec)| (*id, l2_distance(query_vector, vec)))
                .collect();
             
            if top_k as usize >= results.len() {
                 results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            } else {
                 results.select_nth_unstable_by(top_k as usize, |a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                 results.truncate(top_k as usize);
                 results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            }
            return results.into_iter().map(|(id, d)| SearchResult{id, distance: d}).collect();
        }


        let mut centroid_dists: Vec<(usize, f64)> = index.centroids.iter()
            .enumerate()
            .map(|(i, c)| (i, l2_distance(query_vector, c)))
            .collect();
        
        let probes = std::cmp::min(N_PROBE, centroid_dists.len());
        centroid_dists.select_nth_unstable_by(probes, |a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        
        let candidates_indices: Vec<usize> = centroid_dists.iter().take(probes).map(|(i, _)| *i).collect();

        let k = top_k as usize;
        
        let top_k_heap = candidates_indices.into_par_iter()
            .fold(
                || BinaryHeap::with_capacity(k + 1),
                |mut heap: BinaryHeap<HeapItem>, idx| {
                    let list = &index.lists[idx];
                    let vectors = &list.vectors;
                    let ids = &list.ids;
                    let n = ids.len();
                    
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

        final_heap.into_sorted_vec().into_iter()
            .map(|item| SearchResult { id: item.id, distance: item.distance.0 })
            .collect()
    }
}
