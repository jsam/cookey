use crate::inline::{log2, reverse_bits_fast};
use crate::traits::TransformElement;

/// Scaling convention for transforms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scaling {
    None,
    /// FFT scaling: 1/√N on both forward and inverse transforms
    Unitary,  
    /// NTT scaling: No scaling on forward, 1/N on inverse
    Standard,  
}

// Core transformation (butterfly operation) for forward and inverse transforms.
fn transform_core<'field, T: TransformElement<'field>>(
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
        for x in input.iter_mut() {
            *x = x.clone() * scale.clone();
        }
    }

    // Perform bit-reversal permutation
    for i in 0..n {
        let j = reverse_bits_fast(i, log2(n));
        if i < j {
            input.swap(i, j);
        }
    }
    

    // Iterative transform computation
    let mut step = 1;
    while step < n {
        let jump = step * 2;
        let root = if inverse {
            T::get_inverse_root_of_unity(jump, field)
        } else {
            T::get_root_of_unity(jump, field)
        };

        for group in (0..n).step_by(jump) {
            let mut omega = T::one(field);
            for k in group..group + step {
                let t = omega.clone() * input[k + step].clone();
                input[k + step] = input[k].clone() - t.clone();
                input[k] = input[k].clone() + t;
                omega = omega * root.clone();
            }
        }
        step *= 2;
    }

    // Apply inverse scaling
    if inverse {
        let scale = match scaling {
            Scaling::None => { return; },
            Scaling::Unitary => T::get_forward_scale_factor(n, field),  // 1/√N
            Scaling::Standard => T::get_scale_factor(n, field), // 1/N    
        };
        
        // Process elements in chunks to minimize scale factor computations
        for chunk in input.chunks_mut(8) {
            for x in chunk {
                *x = x.clone() * scale.clone();
            }
        }
    }
}

/// Performs an in-place forward transformation.
pub fn cooley_tukey_forward_iterative<'field, T: TransformElement<'field>>(
    input: &mut [T], 
    scaling: Scaling,
    field: &'field T::FieldContext) 
{
    transform_core(input, false, scaling, field);
}

/// Performs an in-place inverse transformation.
pub fn cooley_tukey_inverse_iterative<'field, T: TransformElement<'field>>(
    input: &mut [T],
    scaling: Scaling, 
    field: &'field T::FieldContext) 
{
    transform_core(input, true, scaling, field);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fields::complex::ComplexField;
    use crate::fields::finite::FiniteField;
    use approx::assert_relative_eq;

    const EPSILON: f64 = 1e-10;

    #[test]
    fn test_complex_forward_inverse() {
        let field = ComplexField::new();

        // Create input using builder
        let input = field.inputs()
            .extend([(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)])
            .build();
        
        // Make a copy for transform
        let mut transformed = input.clone();
        cooley_tukey_forward_iterative(&mut transformed, Scaling::Unitary, &field);
        
        // Apply inverse transform
        cooley_tukey_inverse_iterative(&mut transformed, Scaling::Unitary, &field);
        
        // Check if we got back the original values
        for (orig, trans) in input.iter().zip(transformed.iter()) {
            assert_relative_eq!(orig.real, trans.real, epsilon = EPSILON);
            assert_relative_eq!(orig.imag, trans.imag, epsilon = EPSILON);
        }
    }

    #[test]
    fn test_complex_known_sequence() {
        let field = ComplexField::new();
        
        // Test with [(1, 0), (0, 0), (0, 0), (0, 0)] which should give constant Fourier coefficients
        let mut input = field.inputs()
            .push((1.0, 0.0))
            .extend([(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)])
            .build();
        
        cooley_tukey_forward_iterative(&mut input, Scaling::Unitary, &field);
        
        // All coefficients should have magnitude 1.0f (unscaled FFT, otherwise 0.25)
        for value in input.iter() {
            assert_relative_eq!(value.magnitude(), 0.5, epsilon = EPSILON);
        }
    }

    #[test]
    fn test_complex_linearity() {
        let field = ComplexField::new();
        
        // Test vectors using builder
        let mut v1 = field.inputs()
            .extend([(1.0, 0.0), (2.0, 0.0)])
            .build();
        
        let mut v2 = field.inputs()
            .extend([(3.0, 0.0), (4.0, 0.0)])
            .build();
        
        let mut sum = field.inputs()
            .extend([(4.0, 0.0), (6.0, 0.0)])  // 1+3, 2+4
            .build();
        
        // Transform individual vectors
        cooley_tukey_forward_iterative(&mut v1, Scaling::Unitary, &field);
        cooley_tukey_forward_iterative(&mut v2, Scaling::Unitary, &field);
        cooley_tukey_forward_iterative(&mut sum, Scaling::Unitary, &field);
        
        // Check linearity: F(v1 + v2) = F(v1) + F(v2)
        for ((x1, x2), s) in v1.iter().zip(v2.iter()).zip(sum.iter()) {
            let sum_transform = x1.clone() + x2.clone();
            assert_relative_eq!(sum_transform.real, s.real, epsilon = EPSILON);
            assert_relative_eq!(sum_transform.imag, s.imag, epsilon = EPSILON);
        }
    }

    #[test]
    fn test_finite_forward_inverse() {
        let field = FiniteField::new(17, 3, 16); // modulus 17, primitive root 3, order 16
        
        // Create test vector using builder
        let input = field.inputs()
            .extend([1, 2, 3, 4])
            .build();
        
        // Make a copy for transform
        let mut transformed = input.clone();
        cooley_tukey_forward_iterative(&mut transformed, Scaling::Standard, &field);
        
        // Apply inverse transform
        cooley_tukey_inverse_iterative(&mut transformed, Scaling::Standard, &field);
        
        // Check if we got back the original values
        for (orig, trans) in input.iter().zip(transformed.iter()) {
            assert_eq!(orig.value, trans.value);
        }
    }

    #[test]
    fn test_finite_known_sequence() {
        let field = FiniteField::new(17, 3, 16);
        
        // Test with [1, 0, 0, 0]
        let mut input = field.inputs()
            .push(1)
            .extend([0, 0, 0])
            .build();
        
        cooley_tukey_forward_iterative(&mut input, Scaling::Standard, &field);
        
        // In finite field, all coefficients should be equal (1 mod 17)
        for value in input.iter() {
            assert_eq!(value.value, 1);
        }
    }

    #[test]
    fn test_finite_linearity() {
        let field = FiniteField::new(17, 3, 16);
        
        // Create test vectors using builder
        let mut v1 = field.inputs()
            .extend([1, 2])
            .build();
        
        let mut v2 = field.inputs()
            .extend([3, 4])
            .build();
        
        let mut sum = field.inputs()
            .extend([4, 6])  // (1+3, 2+4) mod 17
            .build();
        
        // Transform vectors
        cooley_tukey_forward_iterative(&mut v1, Scaling::Standard, &field);
        cooley_tukey_forward_iterative(&mut v2, Scaling::Standard, &field);
        cooley_tukey_forward_iterative(&mut sum, Scaling::Standard, &field);
        
        // Check linearity in finite field
        for ((x1, x2), s) in v1.iter().zip(v2.iter()).zip(sum.iter()) {
            let sum_transform = x1.clone() + x2.clone();
            assert_eq!(sum_transform.value, s.value);
        }
    }

    #[test]
    fn test_complex_various_sizes() {
        let field = ComplexField::new();
        
        // Test powers of 2 from 2 to 16
        for power in 1..=4 {
            let size = 1 << power;
            
            // Create input of specified size using builder
            let mut input = field.inputs();
            for i in 0..size {
                input.push((i as f64, 0.0));
            }
            let mut input = input.build();
            
            let original = input.clone();
            
            // Forward and inverse transform
            cooley_tukey_forward_iterative(&mut input, Scaling::Unitary, &field);
            cooley_tukey_inverse_iterative(&mut input, Scaling::Unitary, &field);
            
            // Check results
            for (orig, result) in original.iter().zip(input.iter()) {
                assert_relative_eq!(orig.real, result.real, epsilon = EPSILON);
                assert_relative_eq!(orig.imag, result.imag, epsilon = EPSILON);
            }
        }
    }

    #[test]
    fn test_finite_various_sizes() {
        let field = FiniteField::new(17, 3, 16);
        
        // Test powers of 2 from 2 to 16
        for power in 1..=4 {
            let size = 1 << power;
            
            // Create input of specified size using builder
            let mut input = field.inputs();
            for i in 0..size {
                input.push(i as u64);
            }
            let mut input = input.build();
            
            let original = input.clone();
            
            // Forward and inverse transform
            cooley_tukey_forward_iterative(&mut input, Scaling::Standard, &field);
            cooley_tukey_inverse_iterative(&mut input, Scaling::Standard, &field);
            
            // Check results
            for (orig, result) in original.iter().zip(input.iter()) {
                assert_eq!(orig.value, result.value);
            }
        }
    }

    #[test]
    #[should_panic(expected = "Input length must be a power of 2")]
    fn test_non_power_of_two_complex() {
        let field = ComplexField::new();
        let mut input = field.inputs()
            .extend([(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)])  // Length 3
            .build();
        
        cooley_tukey_forward_iterative(&mut input, Scaling::Unitary, &field);
    }

    #[test]
    #[should_panic(expected = "Input length must be a power of 2")]
    fn test_non_power_of_two_finite() {
        let field = FiniteField::new(17, 3, 16);
        let mut input = field.inputs()
            .extend([1, 2, 3])  // Length 3
            .build();
        
        cooley_tukey_forward_iterative(&mut input, Scaling::Standard, &field);
    }

    #[test]
    fn test_complex_numerical_stability() {
        let field = ComplexField::new();
        
        // Test with small values
        let mut input = field.inputs();
        for _ in 0..4 {
            input.push((1e-7, 0.0));
        }
        let mut small = input.build();
        
        let original = small.clone();
        
        // Forward and inverse transform
        cooley_tukey_forward_iterative(&mut small, Scaling::Unitary, &field);
        cooley_tukey_inverse_iterative(&mut small, Scaling::Unitary, &field);
        
        // Check relative error
        for (orig, result) in original.iter().zip(small.iter()) {
            assert_relative_eq!(orig.real, result.real, epsilon = 1e-6);
            assert_relative_eq!(orig.imag, result.imag, epsilon = 1e-6);
        }
        
        // Test with large values
        let mut input = field.inputs();
        for _ in 0..4 {
            input.push((1e7, 0.0));
        }
        let mut large = input.build();
        
        let original = large.clone();
        
        // Forward and inverse transform
        cooley_tukey_forward_iterative(&mut large, Scaling::Unitary, &field);
        cooley_tukey_inverse_iterative(&mut large, Scaling::Unitary, &field);
        
        // Check relative error
        for (orig, result) in original.iter().zip(large.iter()) {
            assert_relative_eq!(orig.real, result.real, epsilon = 1e-6);
            assert_relative_eq!(orig.imag, result.imag, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_convolution_property() {
        let field = ComplexField::new();
        
        // Test vectors [1, 2] and [3, 4]
        let mut v1 = field.inputs()
            .extend([(1.0, 0.0), (2.0, 0.0)])
            .build();
            
        let mut v2 = field.inputs()
            .extend([(3.0, 0.0), (4.0, 0.0)])
            .build();
        
        // Transform both vectors
        cooley_tukey_forward_iterative(&mut v1, Scaling::Standard, &field);
        cooley_tukey_forward_iterative(&mut v2, Scaling::Standard, &field);
        
        // Pointwise multiplication in frequency domain
        let mut conv = v1.iter().zip(v2.iter())
            .map(|(x, y)| x.clone() * y.clone())
            .collect::<Vec<_>>();

        // Inverse transform
        cooley_tukey_inverse_iterative(&mut conv, Scaling::Standard, &field);
        
        // NOTE: Check if circular convolution is done.
        // First element should be (21+1) / 2 = 11.0
        assert_relative_eq!(conv[0].real, 11.0, epsilon = EPSILON);
        assert_relative_eq!(conv[0].imag, 0.0, epsilon = EPSILON);
        
        // Second element should be (21-1) / 2 = 10.0
        assert_relative_eq!(conv[1].real, 10.0, epsilon = EPSILON);
        assert_relative_eq!(conv[1].imag, 0.0, epsilon = EPSILON);
    }

    #[test]
    fn test_padding_transform() {
        let field = ComplexField::new();
        
        // Create a vector of non-power-of-2 length and pad it
        let mut input = field.inputs()
            .extend([(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)])
            .pad_to_power_of_two()
            .build();
        
        // Should now be length 4
        assert_eq!(input.len(), 4);
        
        // Try the transform (should not panic)
        cooley_tukey_forward_iterative(&mut input, Scaling::Unitary, &field);
        cooley_tukey_inverse_iterative(&mut input, Scaling::Unitary, &field);
        
        // Check first three values are preserved
        assert_relative_eq!(input[0].real, 1.0, epsilon = EPSILON);
        assert_relative_eq!(input[1].real, 2.0, epsilon = EPSILON);
        assert_relative_eq!(input[2].real, 3.0, epsilon = EPSILON);
    }

    #[test]
    fn test_builder_chaining() {
        let field = ComplexField::new();
        
        let input = field.inputs()
            .push((1.0, 0.0))
            .extend([(2.0, 0.0), (3.0, 0.0)])
            .pad_zeros(5)
            .build();
        
        assert_eq!(input.len(), 5);
        assert_relative_eq!(input[0].real, 1.0, epsilon = EPSILON);
        assert_relative_eq!(input[3].real, 0.0, epsilon = EPSILON);
        assert_relative_eq!(input[4].real, 0.0, epsilon = EPSILON);
    }

}