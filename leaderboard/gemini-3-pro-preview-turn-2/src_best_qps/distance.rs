pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            (diff * diff) as f64
        })
        .sum::<f64>()
        // .sqrt() // Squared L2 is monotonic, but the spec asks for "distance", usually implies sqrt. 
        // Let's check the API spec. It says "distance: 1.23". Usually that means Euclidean distance.
        // I will return the sqrt just to be safe.
        // Optimization: We can sort by squared distance and only sqrt at the end.
        // But let's follow the standard definition for now.
        .sqrt()
}
