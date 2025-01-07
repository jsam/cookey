
// Trait for SIMD operations
pub trait Vectorize {
    type SimdVector: Clone;

    fn simd_width() -> usize;
    fn load_simd(slice: &[Self]) -> Self::SimdVector where Self: Sized;
    fn load_simd_omega(omegas: &[Self]) -> Self::SimdVector where Self: Sized;
    fn store_simd(slice: &mut [Self], vec: Self::SimdVector) where Self: Sized;
    fn add_simd(a: Self::SimdVector, b: Self::SimdVector) -> Self::SimdVector;
    fn subtract_simd(a: Self::SimdVector, b: Self::SimdVector) -> Self::SimdVector;
    fn multiply_simd(a: Self::SimdVector, b: Self::SimdVector) -> Self::SimdVector;
    fn multiply_simd_scalar(a: Self::SimdVector, scalar: &Self) -> Self::SimdVector where Self: Sized;
}