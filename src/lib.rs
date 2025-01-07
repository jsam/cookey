#![feature(portable_simd)]

pub mod cache;
pub mod fields;
pub mod inline;
pub mod traits;
pub mod transform;
pub mod simd;

// Define optimal alignment for modern CPUs (ie. 32 bytes for AVX2)
pub const ALIGNMENT: usize = 128;
