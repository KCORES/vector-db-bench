#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

pub fn l2_distance(a: &[f32], b: &[f32]) -> f64 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { l2_distance_avx512(a, b) };
        } else if is_x86_feature_detected!("avx2") {
            return unsafe { l2_distance_avx2(a, b) };
        }
    }
    l2_distance_scalar(a, b)
}

fn l2_distance_scalar(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
     .zip(b.iter())
     .map(|(x, y)| {
         let diff = x - y;
         (diff * diff) as f64
     })
     .sum()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn l2_distance_avx2(a: &[f32], b: &[f32]) -> f64 {
    let mut sum = _mm256_setzero_ps();
    let n = a.len();
    if n == 128 {
        let ptr_a = a.as_ptr();
        let ptr_b = b.as_ptr();
        // 128 / 8 = 16 iterations
        for i in 0..16 {
            let va = _mm256_loadu_ps(ptr_a.add(i*8));
            let vb = _mm256_loadu_ps(ptr_b.add(i*8));
            let diff = _mm256_sub_ps(va, vb);
            let sq = _mm256_mul_ps(diff, diff);
            sum = _mm256_add_ps(sum, sq);
        }
        let mut temp = [0.0f32; 8];
        _mm256_storeu_ps(temp.as_mut_ptr(), sum);
        return temp.iter().sum::<f32>() as f64;
    }

    let mut i = 0;
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(va, vb);
        let sq = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, sq);
        i += 8;
    }

    let mut temp = [0.0f32; 8];
    _mm256_storeu_ps(temp.as_mut_ptr(), sum);
    let mut total = temp.iter().sum::<f32>() as f64;

    while i < n {
        let diff = a[i] - b[i];
        total += (diff * diff) as f64;
        i += 1;
    }
    total
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn l2_distance_avx512(a: &[f32], b: &[f32]) -> f64 {
    // Optimized for 128 dim
    if a.len() == 128 {
        let ptr_a = a.as_ptr();
        let ptr_b = b.as_ptr();
        
        let mut sum0 = _mm512_setzero_ps();
        let mut sum1 = _mm512_setzero_ps();
        let mut sum2 = _mm512_setzero_ps();
        let mut sum3 = _mm512_setzero_ps();
        
        // Unroll 4x (4 * 16 = 64 floats)
        // 128 floats = 2 passes of 4x unroll, or just 1 pass of 8x unroll.
        // Let's do 8x unroll directly since registers are plenty (zmm0-31).
        
        // Group 0
        let va0 = _mm512_loadu_ps(ptr_a);
        let vb0 = _mm512_loadu_ps(ptr_b);
        let diff0 = _mm512_sub_ps(va0, vb0);
        sum0 = _mm512_fmadd_ps(diff0, diff0, sum0);

        let va1 = _mm512_loadu_ps(ptr_a.add(16));
        let vb1 = _mm512_loadu_ps(ptr_b.add(16));
        let diff1 = _mm512_sub_ps(va1, vb1);
        sum1 = _mm512_fmadd_ps(diff1, diff1, sum1);

        let va2 = _mm512_loadu_ps(ptr_a.add(32));
        let vb2 = _mm512_loadu_ps(ptr_b.add(32));
        let diff2 = _mm512_sub_ps(va2, vb2);
        sum2 = _mm512_fmadd_ps(diff2, diff2, sum2);

        let va3 = _mm512_loadu_ps(ptr_a.add(48));
        let vb3 = _mm512_loadu_ps(ptr_b.add(48));
        let diff3 = _mm512_sub_ps(va3, vb3);
        sum3 = _mm512_fmadd_ps(diff3, diff3, sum3);

        let va4 = _mm512_loadu_ps(ptr_a.add(64));
        let vb4 = _mm512_loadu_ps(ptr_b.add(64));
        let diff4 = _mm512_sub_ps(va4, vb4);
        sum0 = _mm512_fmadd_ps(diff4, diff4, sum0);

        let va5 = _mm512_loadu_ps(ptr_a.add(80));
        let vb5 = _mm512_loadu_ps(ptr_b.add(80));
        let diff5 = _mm512_sub_ps(va5, vb5);
        sum1 = _mm512_fmadd_ps(diff5, diff5, sum1);

        let va6 = _mm512_loadu_ps(ptr_a.add(96));
        let vb6 = _mm512_loadu_ps(ptr_b.add(96));
        let diff6 = _mm512_sub_ps(va6, vb6);
        sum2 = _mm512_fmadd_ps(diff6, diff6, sum2);

        let va7 = _mm512_loadu_ps(ptr_a.add(112));
        let vb7 = _mm512_loadu_ps(ptr_b.add(112));
        let diff7 = _mm512_sub_ps(va7, vb7);
        sum3 = _mm512_fmadd_ps(diff7, diff7, sum3);

        // Sum up accumulators
        let sum01 = _mm512_add_ps(sum0, sum1);
        let sum23 = _mm512_add_ps(sum2, sum3);
        let sum = _mm512_add_ps(sum01, sum23);

        return _mm512_reduce_add_ps(sum) as f64;
    }

    let mut sum = _mm512_setzero_ps();
    let n = a.len();
    let mut i = 0;
    
    while i + 16 <= n {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));
        let diff = _mm512_sub_ps(va, vb);
        sum = _mm512_fmadd_ps(diff, diff, sum);
        i += 16;
    }

    let mut total = _mm512_reduce_add_ps(sum) as f64;

    while i < n {
        let diff = a[i] - b[i];
        total += (diff * diff) as f64;
        i += 1;
    }

    total
}
