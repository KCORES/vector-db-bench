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

#[target_feature(enable = "avx512f")]
unsafe fn l2_distance_avx512(a: &[f32], b: &[f32]) -> f64 {
    let mut sum = _mm512_setzero_ps();
    let n = a.len();
    let mut i = 0;

    // Process 16 floats at a time
    while i + 16 <= n {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));
        let diff = _mm512_sub_ps(va, vb);
        sum = _mm512_fmadd_ps(diff, diff, sum);
        i += 16;
    }

    let mut remaining = 0.0;
    if i < n {
         // Create a mask for the remaining elements
         let mask = (1u16 << (n - i)) - 1;
         let va = _mm512_maskz_loadu_ps(mask, a.as_ptr().add(i));
         let vb = _mm512_maskz_loadu_ps(mask, b.as_ptr().add(i));
         let diff = _mm512_sub_ps(va, vb);
         // simple sum for tail since it's masked
         let sq_diff = _mm512_mul_ps(diff, diff);
         remaining = _mm512_reduce_add_ps(sq_diff);
    }
    
    _mm512_reduce_add_ps(sum) as f64 + remaining as f64
}

#[target_feature(enable = "avx2")]
unsafe fn l2_distance_avx2(a: &[f32], b: &[f32]) -> f64 {
    let mut sum = _mm256_setzero_ps();
    let n = a.len();
    let mut i = 0;

    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
        i += 8;
    }

    let mut total = 0.0;
    // reduce AVX register
    let mut out = [0.0; 8];
    _mm256_storeu_ps(out.as_mut_ptr(), sum);
    total += out.iter().sum::<f32>();

    // finish tail
    while i < n {
        let diff = a[i] - b[i];
        total += diff * diff;
        i += 1;
    }

    total as f64
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
