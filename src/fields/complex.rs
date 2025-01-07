use std::ops::{Add, Sub, Mul};
use std::f64::consts::PI;

use crate::traits::{One, Transform, TransformElement, Zero};
use crate::transform::{cooley_tukey_forward_iterative, cooley_tukey_inverse_iterative, Scaling};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComplexField {
    scaling: Scaling
}


impl ComplexField {
    pub fn new() -> Self {
        ComplexField {
            scaling: Scaling::Standard
        }
    }

    pub fn with_scaling(scaling: Scaling) -> Self {
        ComplexField { scaling }
    }
}

impl<'a> Transform<'a> for ComplexField {
    type Element = ComplexElement<'a>;

    fn transform_forward(&'a self, input: &mut [Self::Element]) {
        // For FFT, we use unitary scaling (1/√N on both transforms)
        cooley_tukey_forward_iterative(input, self.scaling, self);
    }

    fn transform_inverse(&'a self, input: &mut [Self::Element]) {
        cooley_tukey_inverse_iterative(input, self.scaling, self);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ComplexElement<'a> {
    pub real: f64,
    pub imag: f64,
    field: &'a ComplexField,
}

impl<'a> ComplexElement<'a> {
    pub fn new(real: f64, imag: f64, field: &'a ComplexField) -> Self {
        ComplexElement { real, imag, field }
    }

    // Helper function to create e^(ix)
    fn exp_i(x: f64) -> (f64, f64) {
        (x.cos(), x.sin())
    }
}

impl<'a> Zero<'a> for ComplexElement<'a> {
    type ZeroContext = ComplexField;
    
    fn zero(field: &'a Self::ZeroContext) -> Self {
        ComplexElement::new(0.0,0.0, field)
    }
    
    fn is_zero(&self) -> bool {
        self.real == 0.0 && self.imag == 0.0
    }

}

impl<'a> One<'a> for ComplexElement<'a> {
    type OneContext = ComplexField;

    fn one(field: &'a Self::OneContext) -> Self {
        ComplexElement::new(1.0,0.0, field)
    }
}

impl<'a> Add for ComplexElement<'a> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        ComplexElement::new(
            self.real + other.real,
            self.imag + other.imag,
            self.field
        )
    }
}

impl<'a> Sub for ComplexElement<'a> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        ComplexElement::new(
            self.real - other.real,
            self.imag - other.imag,
            self.field
        )
    }
}

impl<'a> Mul for ComplexElement<'a> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        ComplexElement::new(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
            self.field
        )
    }
}

impl<'a> TransformElement<'a> for ComplexElement<'a> {
    type FieldContext = ComplexField;

    fn get_root_of_unity(n: usize, field: &'a Self::FieldContext) -> Self {
        // For FFT: exp(-2πi/n)
        let angle = -2.0 * PI / n as f64;
        let (cos, sin) = Self::exp_i(angle);
        ComplexElement::new(cos, sin, field)
    }

    fn get_inverse_root_of_unity(n: usize, field: &'a Self::FieldContext) -> Self {
        // For inverse FFT: exp(2πi/n)
        let angle = 2.0 * PI / n as f64;
        let (cos, sin) = Self::exp_i(angle);
        ComplexElement::new(cos, sin, field)
    }

    fn get_forward_scale_factor(n: usize, field: &'a Self::FieldContext) -> Self {
        let scale = 1.0 / (n as f64).sqrt();  // For FFT uniform scaling: 1/√n
        ComplexElement::new(scale, 0.0, field)
    }

    fn get_scale_factor(n: usize, field: &'a Self::FieldContext) -> Self {
        // For FFT: 1/n
        ComplexElement::new(1.0 / n as f64, 0.0, field)
    }
}

impl<'a> ComplexElement<'a> {
    pub fn magnitude(&self) -> f64 {
        (self.real * self.real + self.imag * self.imag).sqrt()
    }

    pub fn phase(&self) -> f64 {
        self.imag.atan2(self.real)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
    use approx::assert_relative_eq; // Make sure to add approx = "0.5" to your Cargo.toml

    const EPSILON: f64 = 1e-10;

    #[test]
    fn test_complex_creation() {
        let field = ComplexField::new();
        let z = ComplexElement::new(3.0, 4.0, &field);
        assert_relative_eq!(z.real, 3.0, epsilon = EPSILON);
        assert_relative_eq!(z.imag, 4.0, epsilon = EPSILON);
    }

    #[test]
    fn test_complex_addition() {
        let field = ComplexField::new();
        let z1 = ComplexElement::new(1.0, 2.0, &field);
        let z2 = ComplexElement::new(3.0, 4.0, &field);
        let result = z1 + z2;
        assert_relative_eq!(result.real, 4.0, epsilon = EPSILON);
        assert_relative_eq!(result.imag, 6.0, epsilon = EPSILON);
    }

    #[test]
    fn test_complex_subtraction() {
        let field = ComplexField::new();
        let z1 = ComplexElement::new(3.0, 4.0, &field);
        let z2 = ComplexElement::new(1.0, 2.0, &field);
        let result = z1 - z2;
        assert_relative_eq!(result.real, 2.0, epsilon = EPSILON);
        assert_relative_eq!(result.imag, 2.0, epsilon = EPSILON);
    }

    #[test]
    fn test_complex_multiplication() {
        let field = ComplexField::new();
        let z1 = ComplexElement::new(1.0, 2.0, &field);
        let z2 = ComplexElement::new(3.0, 4.0, &field);
        let result = z1 * z2;
        // (1 + 2i)(3 + 4i) = (1*3 - 2*4) + (1*4 + 2*3)i = -5 + 10i
        assert_relative_eq!(result.real, -5.0, epsilon = EPSILON);
        assert_relative_eq!(result.imag, 10.0, epsilon = EPSILON);
    }

    #[test]
    fn test_magnitude_and_phase() {
        let field = ComplexField::new();
        let z = ComplexElement::new(3.0, 4.0, &field);
        assert_relative_eq!(z.magnitude(), 5.0, epsilon = EPSILON);
        assert_relative_eq!(z.phase(), 0.927295218001612, epsilon = EPSILON); // atan2(4/3)
    }

    #[test]
    fn test_root_of_unity() {
        let field = ComplexField::new();
        let n = 4;
        let root = ComplexElement::get_root_of_unity(n, &field);
        
        // For n=4, should be exp(-2πi/4) = -i
        assert_relative_eq!(root.real, 0.0, epsilon = EPSILON);
        assert_relative_eq!(root.imag, -1.0, epsilon = EPSILON);

        // Test that (root)^n = 1
        let mut product = root.clone();
        for _ in 1..n {
            product = product * root.clone();
        }
        assert_relative_eq!(product.real, 1.0, epsilon = EPSILON);
        assert_relative_eq!(product.imag, 0.0, epsilon = EPSILON);
    }

    #[test]
    fn test_inverse_root_of_unity() {
        let field = ComplexField::new();
        let n = 4;
        let root = ComplexElement::get_inverse_root_of_unity(n, &field);
        
        // For n=4, should be exp(2πi/4) = i
        assert_relative_eq!(root.real, 0.0, epsilon = EPSILON);
        assert_relative_eq!(root.imag, 1.0, epsilon = EPSILON);

        // Test that it's the inverse of the forward root
        let forward_root = ComplexElement::get_root_of_unity(n, &field);
        let product = root * forward_root;
        assert_relative_eq!(product.real, 1.0, epsilon = EPSILON);
        assert_relative_eq!(product.imag, 0.0, epsilon = EPSILON);
    }

    #[test]
    fn test_scale_factor() {
        let field = ComplexField::new();
        let n = 8;
        let scale = ComplexElement::get_scale_factor(n, &field);
        
        // Should be 1/n
        assert_relative_eq!(scale.real, 1.0 / n as f64, epsilon = EPSILON);
        assert_relative_eq!(scale.imag, 0.0, epsilon = EPSILON);
    }

    #[test]
    fn test_zero_element() {
        let field = ComplexField::new();
        let z = ComplexElement::new(0.0, 0.0, &field);
        assert!(z.is_zero());
    }

    #[test]
    fn test_roots_of_unity_powers() {
        let field = ComplexField::new();
        let n = 8;
        let root = ComplexElement::get_root_of_unity(n, &field);
        
        let mut current = ComplexElement::new(1.0, 0.0, &field);
        for k in 0..n {
            // Check that each power lies on the unit circle
            assert_relative_eq!(current.magnitude(), 1.0, epsilon = EPSILON);
            
            // Check that the angle is correct
            let expected_angle = -2.0 * PI * k as f64 / n as f64;
            let angle = current.phase();
            // Normalize angles to [-π, π] for comparison
            let normalized_expected = (expected_angle + PI).rem_euclid(2.0 * PI) - PI;
            let normalized_actual = (angle + PI).rem_euclid(2.0 * PI) - PI;
            assert_relative_eq!(normalized_actual, normalized_expected, epsilon = EPSILON);
            
            current = current * root.clone();
        }
        
        // After n iterations, should be back at 1
        assert_relative_eq!(current.real, 1.0, epsilon = EPSILON);
        assert_relative_eq!(current.imag, 0.0, epsilon = EPSILON);
    }

    #[test]
    fn test_zero_creation_with_field() {
        let field = ComplexField::new();
        let z = ComplexElement::zero(&field);
        assert!(z.is_zero());
    }

    #[test]
    fn test_one_creation_with_field() {
        let field = ComplexField::new();
        let z = ComplexElement::one(&field);
        assert_eq!(z.real, 1.0);
        assert_eq!(z.imag, 0.0);
    }

    #[test]
    fn test_complex_transform_wrapper() {
        let field = ComplexField::new();
        
        // Create test input
        let mut input = field.inputs()
            .extend([(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)])
            .build();
        
        let original = input.clone();
        
        // Test forward and inverse transforms using the wrapper
        field.transform_forward(&mut input);
        field.transform_inverse(&mut input);
        
        // Check if we got back the original values
        for (orig, result) in original.iter().zip(input.iter()) {
            assert_relative_eq!(orig.real, result.real, epsilon = EPSILON);
            assert_relative_eq!(orig.imag, result.imag, epsilon = EPSILON);
        }
    }

    #[test]
    fn test_complex_convolution_with_wrapper() {
        let field = ComplexField::with_scaling(Scaling::Standard);
        
        // Create test vectors
        let mut v1 = field.inputs()
            .extend([(1.0, 0.0), (2.0, 0.0)])
            .build();
        
        let mut v2 = field.inputs()
            .extend([(3.0, 0.0), (4.0, 0.0)])
            .build();
        
        // Transform both vectors using wrapper
        field.transform_forward(&mut v1);
        field.transform_forward(&mut v2);
        
        // Pointwise multiplication in frequency domain
        let mut conv = v1.iter().zip(v2.iter())
            .map(|(x, y)| x.clone() * y.clone())
            .collect::<Vec<_>>();
        
        // Inverse transform using wrapper
        field.transform_inverse(&mut conv);
        
        // Check convolution results (same as previous test)
        assert_relative_eq!(conv[0].real, 11.0, epsilon = EPSILON);
        assert_relative_eq!(conv[0].imag, 0.0, epsilon = EPSILON);
        assert_relative_eq!(conv[1].real, 10.0, epsilon = EPSILON);
        assert_relative_eq!(conv[1].imag, 0.0, epsilon = EPSILON);
    }

    #[test]
    fn test_unitary_complex_convolution() {
        let field = ComplexField::with_scaling(Scaling::Unitary);
        
        let mut v1 = field.inputs()
            .extend([(1.0, 0.0), (2.0, 0.0)])
            .build();
        
        let mut v2 = field.inputs()
            .extend([(3.0, 0.0), (4.0, 0.0)])
            .build();
        
        field.transform_forward(&mut v1);
        field.transform_forward(&mut v2);
        
        println!("After forward transform with unitary scaling:");
        println!("v1: {:?}", v1);
        println!("v2: {:?}", v2);
        
        let mut conv = v1.iter().zip(v2.iter())
            .map(|(x, y)| x.clone() * y.clone())
            .collect::<Vec<_>>();
        
        println!("After multiplication:");
        println!("conv: {:?}", conv);
        
        field.transform_inverse(&mut conv);
        
        println!("After inverse transform with unitary scaling:");
        println!("conv: {:?}", conv);
        
        assert_relative_eq!(conv[0].real, 7.77817459305202, epsilon = EPSILON);
        assert_relative_eq!(conv[0].imag, 0.0, epsilon = EPSILON);
        assert_relative_eq!(conv[1].real, 7.071067811865472, epsilon = EPSILON);
        assert_relative_eq!(conv[1].imag, 0.0, epsilon = EPSILON);
    }
}