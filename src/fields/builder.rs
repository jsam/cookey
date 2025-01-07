use aligned_vec::{AVec, ConstAlign};

use crate::{traits::{TransformElement, Zero}, ALIGNMENT};

use super::{complex::{ComplexElement, ComplexField}, finite::{FiniteField, FiniteFieldElement}};


// Simple builder for transform inputs
pub struct TransformBuilder<'a, T: TransformElement<'a>> {
    field: &'a T::FieldContext,
    values: AVec<T, ConstAlign<ALIGNMENT>>,
}

impl<'a, T: TransformElement<'a>> TransformBuilder<'a, T> {
    pub fn new(field: &'a T::FieldContext) -> Self {
        Self {
            values: AVec::new(ALIGNMENT),
            field,
        }
    }

    pub fn with_capacity(field: &'a T::FieldContext, capacity: usize) -> Self {
        Self {
            values: AVec::with_capacity(ALIGNMENT, capacity.next_power_of_two()),
            field,
        }
    }

    pub fn build(&self) -> AVec<T, ConstAlign<ALIGNMENT>> {
        self.values.clone()
    }

    pub fn pad_to_power_of_two(&mut self) -> &mut Self 
    where T: Zero<'a>
    {
        let current_len = self.values.len();
        let target_len = current_len.next_power_of_two();
        self.pad_zeros(target_len)
    }

    pub fn pad_zeros(&mut self, target_len: usize) -> &mut Self 
    where T: Zero<'a>
    {
        while self.values.len() < target_len {
            self.values.push(T::zero(self.field));
        }
        self
    }
}

// Specific implementation for ComplexField
impl<'a> TransformBuilder<'a, ComplexElement<'a>> {
    
    pub fn push(&mut self, complex: (f64, f64)) -> &mut Self {
        self.values.push(ComplexElement::new(complex.0, complex.1, self.field));
        self
    }

    pub fn extend<I: IntoIterator<Item = (f64, f64)>>(&mut self, iter: I) -> &mut Self {
        for cnum in iter {
            self.push(cnum);
        }
        self
    }
}

// Specific implementation for FiniteField
impl<'a> TransformBuilder<'a, FiniteFieldElement<'a>> {
    pub fn push(&mut self, value: u64) -> &mut Self {
        self.values.push(FiniteFieldElement::new(value, self.field));
        self
    }

    pub fn extend<I: IntoIterator<Item = u64>>(&mut self, iter: I) -> &mut Self {
        for value in iter {
            self.push(value);
        }
        self
    }
}

// Add builder methods to field types
impl ComplexField {
    pub fn inputs(&self) -> TransformBuilder<ComplexElement> {
        TransformBuilder::new(self)
    }
}

impl FiniteField {
    pub fn inputs(&self) -> TransformBuilder<FiniteFieldElement> {
        TransformBuilder::new(self)
    }
}

// Example usage
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_builder() {
        let field = ComplexField::new();
        let vec = field.inputs()
            .extend([(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)])
            .push((4.0, 0.0))
            .build();

        assert_eq!(vec.len(), 4);
        assert_eq!(vec[0].real, 1.0);
    }

    #[test]
    fn test_finite_builder() {
        let field = FiniteField::new(17, 3, 16);
        let vec = field.inputs()
            .extend([1, 2, 3])
            .push(4)
            .build();

        assert_eq!(vec.len(), 4);
        assert_eq!(vec[0].value, 1);
    }

    #[test]
    fn test_padding() {
        // Test complex field padding
        let complex_field = ComplexField::new();
        let vec = complex_field.inputs()
            .extend([(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)])
            .pad_to_power_of_two()
            .build();

        assert_eq!(vec.len(), 4); // Next power of 2 after 3
        assert_eq!(vec[3].real, 0.0);
        assert_eq!(vec[3].imag, 0.0);

        // Test finite field padding
        let finite_field = FiniteField::new(17, 3, 16);
        let vec = finite_field.inputs()
            .extend([1, 2, 3, 4, 5])
            .pad_to_power_of_two()
            .build();

        assert_eq!(vec.len(), 8); // Next power of 2 after 5
        assert_eq!(vec[7].value, 0);
    }
}