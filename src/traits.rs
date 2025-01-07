use std::ops::{Add, Mul, Sub};

pub trait Zero<'field>: Sized + Add<Self, Output = Self> {
    type ZeroContext;

    fn zero(field: &'field Self::ZeroContext) -> Self;
    
    fn is_zero(&self) -> bool;

    fn set_zero(&mut self, field: &'field Self::ZeroContext) {
        *self = Zero::zero(field);
    }

}

pub trait One<'field>: Sized + Mul<Self, Output = Self> {
    type OneContext;
    
    fn one(field: &'field Self::OneContext) -> Self;

    #[inline]
    fn is_one(&self, field: &'field Self::OneContext) -> bool
    where
        Self: PartialEq,
    {
        *self == Self::one(field)
    }

    fn set_one(&mut self, field: &'field Self::OneContext) {
        *self = One::one(field);
    }
}


/// Trait for types that can be used with the Cooley-Tukey algorithm
pub trait TransformElement<'field>: 
    Clone + 
    Zero<'field, ZeroContext = Self::FieldContext> +
    One<'field, OneContext = Self::FieldContext> +
    Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> {
    type FieldContext;

    /// Get the primitive root of unity of order n
    /// For FFT: exp(-2πi/n)
    /// For NTT: primitive root of unity modulo p where p is the modulus
    fn get_root_of_unity(n: usize, field: &'field Self::FieldContext) -> Self;
    
    /// Get the inverse of the primitive root of unity of order n
    /// For FFT: exp(2πi/n)
    /// For NTT: modular multiplicative inverse of the root of unity
    fn get_inverse_root_of_unity(n: usize, field: &'field Self::FieldContext) -> Self;

    /// Get the forward scaling factor.    
    fn get_forward_scale_factor(n: usize, field: &'field Self::FieldContext) -> Self;

    /// Get the scaling factor for the inverse transform
    /// For FFT: 1/n
    /// For NTT: modular multiplicative inverse of n
    fn get_scale_factor(n: usize, field: &'field Self::FieldContext) -> Self;
}

/// Trait for fields that support forward and inverse transforms
pub trait Transform<'a> {
    type Element: TransformElement<'a>;
    
    /// Perform forward transform on a slice of field elements
    fn transform_forward(&'a self, input: &mut [Self::Element]);
    
    /// Perform inverse transform on a slice of field elements
    fn transform_inverse(&'a self, input: &mut [Self::Element]);
}
