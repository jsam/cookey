use std::ops::{Add, Mul, Sub};

use crate::{cache::ROOT_CACHE, traits::{One, Transform, TransformElement, Zero}, transform::{cooley_tukey_forward_iterative, cooley_tukey_inverse_iterative, Scaling}};

#[derive(Debug, Clone, PartialEq)]
pub struct FiniteField {
    pub modulus: u64,
    root: u64,       // Primitive root of unity of maximum order
    root_order: u64, // Order of the primitive root
}

impl FiniteField {
    pub fn new(modulus: u64, root: u64, root_order: u64) -> Self {
        // Verify the parameters are valid
        assert!(FiniteFieldElement::pow_mod(root, root_order, modulus) == 1, 
            "Root must have specified order");
        assert!(FiniteFieldElement::pow_mod(root, root_order / 2, modulus) != 1, 
            "Root must be primitive");

        FiniteField {
            modulus,
            root,
            root_order,
        }
    }

    // Get a root of unity of order n
    fn get_root_of_unity(&self, n: usize) -> u64 {
        assert!(n.is_power_of_two(), "n must be a power of 2");
        assert!(n as u64 <= self.root_order, "n must not exceed root order");
        
        let cache = ROOT_CACHE.read().unwrap();
        if let Some(&root) = cache.roots.get(&(self.modulus, self.root, n)) {
            return root;
        }
        drop(cache);
        
        let root = FiniteFieldElement::pow_mod(self.root, self.root_order / n as u64, self.modulus);
        let mut cache = ROOT_CACHE.write().unwrap();
        cache.roots.insert((self.modulus, self.root, n), root);
        root
    }

    // Get the inverse of a root of unity of order n
    fn get_inverse_root_of_unity(&self, n: usize) -> u64 {
        assert!(n.is_power_of_two(), "n must be a power of 2");
        assert!(n as u64 <= self.root_order, "n must not exceed root order");
        
        let cache = ROOT_CACHE.read().unwrap();
        if let Some(&inv_root) = cache.inverse_roots.get(&(self.modulus, self.root, n)) {
            return inv_root;
        }
        drop(cache);
        
        let root_n = self.get_root_of_unity(n);
        let inv_root = FiniteFieldElement::mod_inverse(root_n, self.modulus)
            .expect("Root of unity must have a multiplicative inverse");
            
        let mut cache = ROOT_CACHE.write().unwrap();
        cache.inverse_roots.insert((self.modulus, self.root, n), inv_root);
        inv_root
    }

    // Get the scaling factor (1/n mod p) for the inverse transform
    fn get_scale_factor(&self, n: usize) -> u64 {
        assert!(n.is_power_of_two(), "n must be a power of 2");
        assert!(n as u64 <= self.root_order, "n must not exceed root order");
        
        let cache = ROOT_CACHE.read().unwrap();
        if let Some(&scale) = cache.scale_factors.get(&(self.modulus, n)) {
            return scale;
        }
        drop(cache);
        
        let scale = FiniteFieldElement::mod_inverse(n as u64, self.modulus)
            .expect("n must have a multiplicative inverse");
            
        let mut cache = ROOT_CACHE.write().unwrap();
        cache.scale_factors.insert((self.modulus, n), scale);
        scale
    }
}

impl<'a> Transform<'a> for FiniteField {
    type Element = FiniteFieldElement<'a>;

    fn transform_forward(&'a self, input: &mut [Self::Element]) {
        // For NTT, we use no scaling on forward transform and 1/N on inverse
        cooley_tukey_forward_iterative(input, Scaling::Standard, self);
    }

    fn transform_inverse(&'a self, input: &mut [Self::Element]) {
        cooley_tukey_inverse_iterative(input, Scaling::Standard, self);
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FiniteFieldElement<'a> {
    pub value: u64,
    field: &'a FiniteField,
}


impl<'a> FiniteFieldElement<'a> {
    pub fn new(value: u64, field: &'a FiniteField) -> Self {
        FiniteFieldElement {
            value: value % field.modulus,
            field,
        }
    }

    // Helper function to compute modular multiplication avoiding overflow
    fn mul_mod(a: u64, b: u64, modulus: u64) -> u64 {
        let mut result = 0;
        let mut a = a;
        let mut b = b;
        while b > 0 {
            if b & 1 != 0 {
                result = (result + a) % modulus;
            }
            a = (a << 1) % modulus;
            b >>= 1;
        }
        result
    }

    // Helper function for modular exponentiation
    fn pow_mod(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
        let mut result = 1;
        base = base % modulus;
        while exp > 0 {
            if exp & 1 != 0 {
                result = Self::mul_mod(result, base, modulus);
            }
            base = Self::mul_mod(base, base, modulus);
            exp >>= 1;
        }
        result
    }

    // Helper function to find multiplicative inverse
    fn mod_inverse(a: u64, m: u64) -> Option<u64> {
        let mut t = 0i64;
        let mut newt = 1i64;
        let mut r = m as i64;
        let mut newr = a as i64;

        while newr != 0 {
            let quotient = r / newr;
            (t, newt) = (newt, t - quotient * newt);
            (r, newr) = (newr, r - quotient * newr);
        }

        if r > 1 {
            return None;
        }
        if t < 0 {
            t += m as i64;
        }
        Some(t as u64)
    }
}

impl<'a> Zero<'a> for FiniteFieldElement<'a> {
    type ZeroContext = FiniteField;
    fn zero(field: &'a Self::ZeroContext) -> Self {
        FiniteFieldElement::new(0, field)
    }
    fn is_zero(&self) -> bool {
        self.value == 0
    }
}

impl<'a> One<'a> for FiniteFieldElement<'a> {
    type OneContext = FiniteField;
    fn one(field: &'a Self::OneContext) -> Self {
        FiniteFieldElement::new(1, field)
    }
}

impl<'a> Add for FiniteFieldElement<'a> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        assert_eq!(self.field.modulus, other.field.modulus, "Fields must match");
        FiniteFieldElement::new(
            (self.value + other.value) % self.field.modulus,
            self.field
        )
    }
}

impl<'a> Sub for FiniteFieldElement<'a> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        assert_eq!(self.field.modulus, other.field.modulus, "Fields must match");
        let value = if self.value >= other.value {
            self.value - other.value
        } else {
            self.field.modulus - (other.value - self.value)
        };
        FiniteFieldElement::new(value, self.field)
    }
}

impl<'a> Mul for FiniteFieldElement<'a> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        assert_eq!(self.field.modulus, other.field.modulus, "Fields must match");
        FiniteFieldElement::new(
            Self::mul_mod(self.value, other.value, self.field.modulus),
            self.field
        )
    }
}

impl<'a> TransformElement<'a> for FiniteFieldElement<'a> {
    type FieldContext = FiniteField;

    fn get_root_of_unity(n: usize, field: &'a Self::FieldContext) -> Self {
        FiniteFieldElement::new(field.get_root_of_unity(n), field)
    }

    fn get_inverse_root_of_unity(n: usize, field: &'a Self::FieldContext) -> Self {
        FiniteFieldElement::new(field.get_inverse_root_of_unity(n), field)
    }

    fn get_forward_scale_factor(_: usize, field: &'a Self::FieldContext) -> Self {
        FiniteFieldElement::one(field)  // NOTE: No scaling for forward NTT
    }

    fn get_scale_factor(n: usize, field: &'a Self::FieldContext) -> Self {
        FiniteFieldElement::new(field.get_scale_factor(n), field)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to set up a standard test field
    // Using modulus 17 and root 3 of order 16 as an example prime field
    fn setup_test_field() -> FiniteField {
        FiniteField::new(17, 3, 16)
    }

    #[test]
    fn test_field_creation() {
        let field = setup_test_field();
        assert_eq!(field.modulus, 17);
        assert_eq!(field.root, 3);
        assert_eq!(field.root_order, 16);
        
        // Verify root properties
        assert_eq!(FiniteFieldElement::pow_mod(3, 16, 17), 1);
        assert_ne!(FiniteFieldElement::pow_mod(3, 8, 17), 1);
    }

    #[test]
    #[should_panic(expected = "Root must have specified order")]
    fn test_invalid_root_order() {
        FiniteField::new(17, 3, 8); // 3^8 ≢ 1 (mod 17)
    }

    #[test]
    fn test_element_creation() {
        let field = setup_test_field();
        let element = FiniteFieldElement::new(20, &field); // Should reduce to 3 mod 17
        assert_eq!(element.value, 3);
    }

    #[test]
    fn test_element_addition() {
        let field = setup_test_field();
        let a = FiniteFieldElement::new(15, &field);
        let b = FiniteFieldElement::new(5, &field);
        let result = a + b;
        assert_eq!(result.value, 3); // (15 + 5) mod 17 = 20 mod 17 = 3
    }

    #[test]
    fn test_element_subtraction() {
        let field = setup_test_field();
        let a = FiniteFieldElement::new(5, &field);
        let b = FiniteFieldElement::new(8, &field);
        let result = a - b;
        assert_eq!(result.value, 14); // (5 - 8) mod 17 = -3 mod 17 = 14
    }

    #[test]
    fn test_element_multiplication() {
        let field = setup_test_field();
        let a = FiniteFieldElement::new(5, &field);
        let b = FiniteFieldElement::new(7, &field);
        let result = a * b;
        assert_eq!(result.value, 1); // (5 * 7) mod 17 = 35 mod 17 = 1
    }

    #[test]
    fn test_zero_element() {
        let field = setup_test_field();
        let zero = FiniteFieldElement::new(0, &field);
        assert!(zero.is_zero());
    }

    #[test]
    fn test_zero_creation_with_field() {
        let field = FiniteField::new(17, 3, 16);  // Example: modulus 17, primitive root 3, order 16
        let z = FiniteFieldElement::zero(&field);
        assert!(z.is_zero());
        assert_eq!(z.value, 0);
    }

    #[test]
    fn test_one_creation_with_field() {
        let field = FiniteField::new(17, 3, 16);  // Example: modulus 17, primitive root 3, order 16
        let z = FiniteFieldElement::one(&field);
        assert_eq!(z.value, 1);
    }

    #[test]
    fn test_root_of_unity() {
        let field = setup_test_field();
        let n = 4;
        let root = FiniteFieldElement::get_root_of_unity(n, &field);
        
        // Test that root^n ≡ 1 (mod p)
        let mut product = root;
        for _ in 1..n {
            product = product * root;
        }
        assert_eq!(product.value, 1);
    }

    #[test]
    fn test_inverse_root_of_unity() {
        let field = setup_test_field();
        let n = 4;
        let root = FiniteFieldElement::get_root_of_unity(n, &field);
        let inv_root = FiniteFieldElement::get_inverse_root_of_unity(n, &field);
        
        // Test that root * inv_root ≡ 1 (mod p)
        let product = root * inv_root;
        assert_eq!(product.value, 1);
    }

    #[test]
    fn test_scale_factor() {
        let field = setup_test_field();
        let n = 4;
        let scale = FiniteFieldElement::get_scale_factor(n, &field);
        
        // Test that n * scale ≡ 1 (mod p)
        let n_element = FiniteFieldElement::new(n as u64, &field);
        let product = n_element * scale;
        assert_eq!(product.value, 1);
    }

    #[test]
    fn test_modular_arithmetic_overflow() {
        let field = FiniteField::new(17, 3, 16);
        let a = FiniteFieldElement::new(u64::MAX - 1, &field);
        let b = FiniteFieldElement::new(2, &field);
        
        // Should handle overflow gracefully
        let result = a + b;
        assert!(result.value < field.modulus);
        
        let result = a * b;
        assert!(result.value < field.modulus);
    }

    #[test]
    fn test_mod_inverse() {
        let value = 3u64;
        let modulus = 17u64;
        let inverse = FiniteFieldElement::mod_inverse(value, modulus).unwrap();
        
        // Test that a * a^(-1) ≡ 1 (mod p)
        assert_eq!(FiniteFieldElement::mul_mod(value, inverse, modulus), 1);
    }

    #[test]
    fn test_pow_mod() {
        let base = 3u64;
        let exp = 5u64;
        let modulus = 17u64;
        let result = FiniteFieldElement::pow_mod(base, exp, modulus);
        
        // 3^5 mod 17 = 243 mod 17 = 5
        assert_eq!(result, 5);
    }

    #[test]
    #[should_panic(expected = "n must be a power of 2")]
    fn test_non_power_of_two() {
        let field = setup_test_field();
        FiniteFieldElement::get_root_of_unity(6, &field);
    }

    #[test]
    #[should_panic(expected = "n must not exceed root order")]
    fn test_order_exceeds_root_order() {
        let field = setup_test_field();
        FiniteFieldElement::get_root_of_unity(32, &field);
    }

    #[test]
    fn test_field_operations_associativity() {
        let field = setup_test_field();
        let a = FiniteFieldElement::new(5, &field);
        let b = FiniteFieldElement::new(7, &field);
        let c = FiniteFieldElement::new(11, &field);
        
        // Test (a + b) + c = a + (b + c)
        assert_eq!((a + b) + c, a + (b + c));
        
        // Test (a * b) * c = a * (b * c)
        assert_eq!((a * b) * c, a * (b * c));
    }

    #[test]
    fn test_finite_transform_wrapper() {
        let field = FiniteField::new(17, 3, 16); // modulus 17, primitive root 3, order 16
        
        // Create test input
        let mut input = field.inputs()
            .extend([1, 2, 3, 4])
            .build();
        
        let original = input.clone();
        
        // Test forward and inverse transforms using the wrapper
        field.transform_forward(&mut input);
        field.transform_inverse(&mut input);
        
        // Check if we got back the original values
        for (orig, result) in original.iter().zip(input.iter()) {
            assert_eq!(orig.value, result.value);
        }
    }

    #[test]
    fn test_finite_convolution_with_wrapper() {
        let field = FiniteField::new(17, 3, 16);
        
        // Create test vectors
        let mut v1 = field.inputs()
            .extend([1, 2])
            .build();
        
        let mut v2 = field.inputs()
            .extend([3, 4])
            .build();
        
        // Transform both vectors using wrapper
        field.transform_forward(&mut v1);
        field.transform_forward(&mut v2);
        
        // Pointwise multiplication in transform domain
        let mut conv = v1.iter().zip(v2.iter())
            .map(|(x, y)| x.clone() * y.clone())
            .collect::<Vec<_>>();
        
        // Inverse transform using wrapper
        field.transform_inverse(&mut conv);
        
        // Values should be computed modulo 17
        assert_eq!(conv[0].value, 11); // (11 mod 17)
        assert_eq!(conv[1].value, 10); // (10 mod 17)
    }

    #[test]
    fn test_root_caching() {
        let field = FiniteField::new(17, 3, 16);
        
        // Clear cache for testing
        ROOT_CACHE.write().unwrap().roots.clear();
        
        // First call - should compute and cache
        let root_8 = field.get_root_of_unity(8);
        assert!(ROOT_CACHE.read().unwrap().roots.contains_key(&(17, 3, 8)));
        
        // Second call - should use cache
        let root_8_cached = field.get_root_of_unity(8);
        assert_eq!(root_8, root_8_cached);
    }

    #[test]
    fn test_concurrent_access() {
        use std::thread;
        
        let _field = FiniteField::new(17, 3, 16);
        
        let threads: Vec<_> = (0..4)
            .map(|_| {
                thread::spawn(move || {
                    let local_field = FiniteField::new(17, 3, 16);
                    local_field.get_root_of_unity(8)
                })
            })
            .collect();
            
        let results: Vec<_> = threads
            .into_iter()
            .map(|t| t.join().unwrap())
            .collect();
            
        assert!(results.windows(2).all(|w| w[0] == w[1]));
    }
}