use crate::api::*;
use std::sync::RwLock;
use rand::Rng;
use rayon::prelude::*;

const DIM: usize = 128;
const MIN_Vectors_FOR_INDEX: usize = 10_000;
const K_FACTOR: usize = 40; // rough guess, N/K_FACTOR for centroid count? No, sqrt(N) is better.

pub struct VectorDB {
    storage: RwLock<Storage>,
}

enum Storage {
    Flat(FlatIndex),
    Ivf(IvfIndex),
}

struct FlatIndex {
    ids: Vec<u64>,
    vectors: Vec<f32>,
    count: usize,
}

struct IvfIndex {
    centroids: Vec<f32>,      // k * DIM
    lists: Vec<FlatIndex>,    // k lists
    num_centroids: usize,
    unindexed: FlatIndex,     // small buffer for new inserts
}

impl FlatIndex {
    fn new() -> Self {
        FlatIndex {
            ids: Vec::new(),
            vectors: Vec::new(),
            count: 0,
        }
    }

    fn push(&mut self, id: u64, vector: &[f32]) {
        self.ids.push(id);
        self.vectors.extend_from_slice(vector);
        self.count += 1;
    }
}

impl VectorDB {
    pub fn new() -> Self {
        VectorDB {
            storage: RwLock::new(Storage::Flat(FlatIndex::new())),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        if vector.len() != DIM { return; }
        
        let mut storage = self.storage.write().unwrap();
        match &mut *storage {
            Storage::Flat(flat) => {
                flat.push(id, &vector);
            }
            Storage::Ivf(ivf) => {
                // Determine closest centroid
                let (closest_idx, _) = find_closest_centroid(&ivf.centroids, &vector);
                ivf.lists[closest_idx].push(id, &vector);
            }
        }
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let mut storage = self.storage.write().unwrap();
        let count = vectors.len();
        
        // Temporary buffer to hold data before we decide where it goes
        // Getting ownership of vectors is good.
        
        match &mut *storage {
            Storage::Flat(flat) => {
                for (id, vec) in vectors {
                    if vec.len() == DIM {
                        flat.push(id, &vec);
                    }
                }
                
                // Check if we should index
                if flat.count >= MIN_Vectors_FOR_INDEX {
                    // Upgrade to IVF
                    // Taking ownership of the data in flat
                    let old_ids = std::mem::take(&mut flat.ids);
                    let old_vecs = std::mem::take(&mut flat.vectors);
                    
                    // Re-structure data for training
                    // We need a list of slices or just use the flat vectors
                    let num_vectors = old_ids.len();
                    let k = (num_vectors as f64).sqrt() as usize; 
                    // limit k? 1M -> 1000.
                    
                    let new_ivf = build_ivf(old_ids, old_vecs, k);
                    *storage = Storage::Ivf(new_ivf);
                }
            }
            Storage::Ivf(ivf) => {
                // For bulk insert into IVF, we can parallelize the assignment
                // But we need to update the lists, which is mutable.
                // Simple way: sequentially insert.
                // Optimized way: pre-calculate assignments in parallel, then group.
                
                // Let's do sequential for now to be safe, or optimize if needed.
                // Given the benchmark phase is usually read-heavy, slow bulk insert is OK-ish,
                // but let's not be too slow.
                for (id, vec) in vectors {
                    if vec.len() == DIM {
                         let (closest_idx, _) = find_closest_centroid(&ivf.centroids, &vec);
                        ivf.lists[closest_idx].push(id, &vec);
                    }
                }
            }
        }
        count
    }

    pub fn search(&self, vector: &[f32], top_k: u32) -> Vec<SearchResult> {
        if vector.len() != DIM { return Vec::new(); }
        
        let storage = self.storage.read().unwrap();
        match &*storage {
            Storage::Flat(flat) => {
                search_flat(flat, vector, top_k)
            }
            Storage::Ivf(ivf) => {
                search_ivf(ivf, vector, top_k)
            }
        }
    }
}

fn search_flat(data: &FlatIndex, vector: &[f32], top_k: u32) -> Vec<SearchResult> {
    // Same logic as before
    let n = data.count;
    let k = top_k as usize;
    let mut candidates: Vec<SearchResult> = Vec::with_capacity(k + 1);
    let vectors_ptr = data.vectors.as_ptr();
    let ids = &data.ids;

    for i in 0..n {
        let dist = unsafe {
             crate::distance::l2_distance(std::slice::from_raw_parts(vectors_ptr.add(i * DIM), DIM), vector)
        };

        if candidates.len() < k {
            candidates.push(SearchResult { id: ids[i], distance: dist });
            if candidates.len() == k {
                candidates.sort_unstable_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
            }
        } else {
            if dist < candidates[k - 1].distance {
                 let idx = candidates.partition_point(|x| x.distance <= dist);
                 candidates.insert(idx, SearchResult { id: ids[i], distance: dist });
                 candidates.truncate(k);
            }
        }
    }
    
    // Final sort to be sure
    if candidates.len() < k {
        candidates.sort_unstable_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
    }
    candidates
}

fn search_ivf(ivf: &IvfIndex, vector: &[f32], top_k: u32) -> Vec<SearchResult> {
    let k = top_k as usize;
    // 1. Find closest centroids
    // We want 'nprobe' closest clusters.
    // Heuristic: nprobe = sqrt(num_centroids) or fixed. 
    // With 1000 centroids, let's try 15.
    let nprobe = 16;
    
    // Scan all centroids
    let mut centroid_dists: Vec<(usize, f64)> = Vec::with_capacity(ivf.num_centroids);
    let c_ptr = ivf.centroids.as_ptr();
    for i in 0..ivf.num_centroids {
         let dist = unsafe {
             crate::distance::l2_distance(std::slice::from_raw_parts(c_ptr.add(i * DIM), DIM), vector)
        };
        centroid_dists.push((i, dist));
    }

    // Pick top nprobe
    centroid_dists.select_nth_unstable_by(nprobe, |a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    
    // Now search locally in these clusters
    // We can reuse the same candidate logic
    let mut candidates: Vec<SearchResult> = Vec::with_capacity(k + 1);

    for (c_idx, _) in centroid_dists.iter().take(nprobe) {
        let list = &ivf.lists[*c_idx];
        let n = list.count;
        if n == 0 { continue; }
        
        let vectors_ptr = list.vectors.as_ptr();
        let ids = &list.ids;
        
        for i in 0..n {
            let dist = unsafe {
                 crate::distance::l2_distance(std::slice::from_raw_parts(vectors_ptr.add(i * DIM), DIM), vector)
            };
            
            // Standard maintain-top-k logic
            if candidates.len() < k {
                candidates.push(SearchResult { id: ids[i], distance: dist });
                if candidates.len() == k {
                    candidates.sort_unstable_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
                }
            } else {
                if dist < candidates[k - 1].distance {
                     let idx = candidates.partition_point(|x| x.distance <= dist);
                     candidates.insert(idx, SearchResult { id: ids[i], distance: dist });
                     candidates.truncate(k);
                }
            }
        }
    }

    // Also search unindexed buffer if any
    {
        let list = &ivf.unindexed;
        if list.count > 0 {
             let vectors_ptr = list.vectors.as_ptr();
             let ids = &list.ids;
             for i in 0..list.count {
                  let dist = unsafe {
                     crate::distance::l2_distance(std::slice::from_raw_parts(vectors_ptr.add(i * DIM), DIM), vector)
                  };
                  if candidates.len() < k {
                    candidates.push(SearchResult { id: ids[i], distance: dist });
                    if candidates.len() == k {
                        candidates.sort_unstable_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
                    }
                } else {
                    if dist < candidates[k - 1].distance {
                         let idx = candidates.partition_point(|x| x.distance <= dist);
                         candidates.insert(idx, SearchResult { id: ids[i], distance: dist });
                         candidates.truncate(k);
                    }
                }
             }
        }
    }

    if candidates.len() < k {
        candidates.sort_unstable_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
    }
    
    candidates
}

fn find_closest_centroid(centroids: &[f32], vector: &[f32]) -> (usize, f64) {
    let num_centroids = centroids.len() / DIM;
    let mut min_dist = f64::MAX;
    let mut min_idx = 0;
    
    let c_ptr = centroids.as_ptr();
    for i in 0..num_centroids {
         let dist = unsafe {
             crate::distance::l2_distance(std::slice::from_raw_parts(c_ptr.add(i * DIM), DIM), vector)
        };
        if dist < min_dist {
            min_dist = dist;
            min_idx = i;
        }
    }
    (min_idx, min_dist)
}

fn build_ivf(ids: Vec<u64>, vectors: Vec<f32>, k: usize) -> IvfIndex {
    let num_vectors = ids.len();
    
    // 1. Initialize centroids (random sample)
    let mut rng = rand::thread_rng();
    let mut centroids = Vec::with_capacity(k * DIM);
    for _ in 0..k {
        let idx = rng.gen_range(0..num_vectors);
        let start = idx * DIM;
        centroids.extend_from_slice(&vectors[start..start + DIM]);
    }
    
    // 2. K-means iterations
    // Use smaller iterations for speed. 3-5 is usually enough for coarse index.
    let iterations = 5;
    for _iter in 0..iterations {
        // Assign points to clusters
        // We need thread-local accumulators for computing new centroids
        
        // Parallel assignment
        // Create atomic or thread-local storage for cluster sums and counts
        // Since k is small (1000), we can just have each thread reduce to a k-sized accumulator
        
        // Using Rayon `fold` + `reduce`
        // Chunks of vectors
        let chunk_size = 1000; // process 1000 vectors at a time
        
        let (new_centroids_acc, counts) = (0..num_vectors)
            .into_par_iter()
            .chunks(chunk_size)
            .map(|chunk_indices| {
                let mut local_sums = vec![0.0f32; k * DIM];
                let mut local_counts = vec![0usize; k];
                
                for i in chunk_indices {
                    let v = &vectors[i * DIM..(i + 1) * DIM];
                    let (best_idx, _) = find_closest_centroid(&centroids, v);
                    
                    local_counts[best_idx] += 1;
                    let sum_start = best_idx * DIM;
                    let sum_slice = &mut local_sums[sum_start..sum_start + DIM];
                    for j in 0..DIM {
                        sum_slice[j] += v[j];
                    }
                }
                (local_sums, local_counts)
            })
            .reduce(
                || (vec![0.0f32; k * DIM], vec![0usize; k]),
                |(mut sums_a, mut counts_a), (sums_b, counts_b)| {
                    // vector add
                    for i in 0..sums_a.len() {
                        sums_a[i] += sums_b[i];
                    }
                    for i in 0..counts_a.len() {
                        counts_a[i] += counts_b[i];
                    }
                    (sums_a, counts_a)
                }
            );

        // Update centroids
        let mut max_move = 0.0;
        for i in 0..k {
            let count = counts[i];
            if count > 0 {
                let start = i * DIM;
                for j in 0..DIM {
                    let new_val = new_centroids_acc[start + j] / count as f32;
                    // centroids[start + j] = new_val; // We can check convergence if we want
                     centroids[start + j] = new_val;
                }
            } else {
                // Re-initialize empty cluster?
                // Pick a random vector again? 
                // Ignore for now.
            }
        }
    }

    // 3. Final assignment and partitioning
    // Create the lists
    let mut lists: Vec<FlatIndex> = (0..k).map(|_| FlatIndex::new()).collect();
    
    // We can't trivially parallelize pushing to lists without locks.
    // But we can compute assignments in parallel: `Vec<usize>` (cluster_id for each vector)
    // Then shuffle data.
    
    let assignments: Vec<usize> = (0..num_vectors)
        .into_par_iter()
        .map(|i| {
            let v = &vectors[i * DIM..(i + 1) * DIM];
            let (best_idx, _) = find_closest_centroid(&centroids, v);
            best_idx
        })
        .collect();

    // Now single-threaded partitioning (memory bandwidth bound mostly)
    for (i, &cluster_idx) in assignments.iter().enumerate() {
        let v = &vectors[i * DIM..(i + 1) * DIM];
        lists[cluster_idx].push(ids[i], v);
    }
    
    IvfIndex {
        centroids,
        lists,
        num_centroids: k,
        unindexed: FlatIndex::new(),
    }
}
