use crate::inline::{log2, reverse_bits_fast};
use crate::traits::TransformElement;
use crate::transform::Scaling;

use super::traits::Vectorize;

pub fn transform_core_simd<'field, T: TransformElement<'field> + Vectorize>(
    input: &mut [T], 
    inverse: bool,
    scaling: Scaling,
    field: &'field T::FieldContext
) {
    let n = input.len();
    if !n.is_power_of_two() {
        panic!("Input length must be a power of 2");
    }

    // Apply forward scaling
    if !inverse && matches!(scaling, Scaling::Unitary) {
        let scale = T::get_forward_scale_factor(n, field);
        simd_scale(input, &scale);
    }

    // Perform bit-reversal permutation
    for i in 0..n {
        let j = reverse_bits_fast(i, log2(n));
        if i < j {
            input.swap(i, j);
        }
    }

    // Iterative transform computation with SIMD
    let mut step = 1;
    while step < n {
        let jump = step * 2;
        let root = if inverse {
            T::get_inverse_root_of_unity(jump, field)
        } else {
            T::get_root_of_unity(jump, field)
        };

        // Process groups in parallel where possible
        for group in (0..n).step_by(jump) {
            let simd_width = T::simd_width();
            let mut k = group;
            
            // Vector for storing omega values
            let mut omegas = Vec::with_capacity(simd_width);
            let mut omega = T::one(field);
            
            while k < group + step {
                // Process SIMD-width elements at once
                if k + simd_width <= group + step {
                    // Prepare omega values
                    omegas.clear();
                    for _ in 0..simd_width {
                        omegas.push(omega.clone());
                        omega = omega * root.clone();
                    }
                    
                    // Load SIMD vectors
                    let mut vec_a = T::load_simd(&input[k..k + simd_width]);
                    let mut vec_b = T::load_simd(&input[k + step..k + step + simd_width]);
                    let vec_omega = T::load_simd_omega(&omegas);
                    
                    // Perform butterfly operation
                    let vec_t = T::multiply_simd(vec_omega, vec_b);
                    vec_b = T::subtract_simd(vec_a.clone(), vec_t.clone());
                    vec_a = T::add_simd(vec_a, vec_t);
                    
                    // Store results
                    T::store_simd(&mut input[k..k + simd_width], vec_a);
                    T::store_simd(&mut input[k + step..k + step + simd_width], vec_b);
                    
                    k += simd_width;
                } else {
                    // Handle remaining elements scalar
                    let t = omega.clone() * input[k + step].clone();
                    input[k + step] = input[k].clone() - t.clone();
                    input[k] = input[k].clone() + t;
                    omega = omega * root.clone();
                    k += 1;
                }
            }
        }
        step *= 2;
    }

    // Apply inverse scaling
    if inverse {
        let scale = match scaling {
            Scaling::None => return,
            Scaling::Unitary => T::get_forward_scale_factor(n, field),
            Scaling::Standard => T::get_scale_factor(n, field),
        };
        simd_scale(input, &scale);
    }
}

#[inline]
pub fn simd_scale<'field, T: TransformElement<'field> + Vectorize>(
    input: &mut [T],
    scale: &T
) {
    let simd_width = T::simd_width();
    let (chunks, remainder) = input.split_at_mut(input.len() - input.len() % simd_width);
    
    // Process main chunks with SIMD
    for chunk in chunks.chunks_mut(simd_width) {
        let vec = T::load_simd(chunk);
        let scaled = T::multiply_simd_scalar(vec, scale);
        T::store_simd(chunk, scaled);
    }
    
    // Handle remaining elements
    for x in remainder {
        *x = x.clone() * scale.clone();
    }
}
