pub mod cache;
pub mod fields;
pub mod inline;
pub mod traits;
pub mod transform;
#[cfg(feature = "simd")]
pub mod simd;

#[cfg(not(feature = "simd"))]
mod simd {
    // dummy module if SIMD is not enabled.
    pub trait Vectorize {}
}

// Define optimal alignment for modern CPUs (ie. 32 bytes for AVX2)
pub const ALIGNMENT: usize = 128;
