use std::simd::{cmp::SimdPartialOrd, u64x4, Mask};


pub fn mod_mul_simd(a: u64x4, b: u64x4, modulus: u64) -> u64x4 {
    let k = 64;
    // Using division instead of shift to avoid overflow
    let r = (!0u128 / modulus as u128) as u64;
    let a_hi = a >> 32;
    let a_lo = a & u64x4::splat(0xFFFFFFFF);
    let b_hi = b >> 32;
    let b_lo = b & u64x4::splat(0xFFFFFFFF);
    let mul_lo = a_lo * b_lo;
    let mul_mid = a_lo * b_hi + a_hi * b_lo;
    let mul_hi = a_hi * b_hi;
    let prod_lo = mul_lo + ((mul_mid & u64x4::splat(0xFFFFFFFF)) << 32);
    let prod_hi = mul_hi + (mul_mid >> 32);
    let q1 = prod_hi >> (k - 1);
    let q2 = mul_hi_word(q1, r);
    let q3 = q2 >> (k + 1);
    let r1 = prod_lo - q3 * u64x4::splat(modulus);
    let modulus_vec = u64x4::splat(modulus);
    r1.simd_ge(modulus_vec).select(r1 - modulus_vec, r1)
}

#[inline]
pub fn mul_hi_word(a: u64x4, b: u64) -> u64x4 {
    let a_hi = a >> 32;
    let a_lo = a & u64x4::splat(0xFFFFFFFF);
    let b_hi = b >> 32;
    let b_lo = b & 0xFFFFFFFF;
    
    let mul_mid = a_lo * u64x4::splat(b_hi) + a_hi * u64x4::splat(b_lo);
    let mul_hi = a_hi * u64x4::splat(b_hi);
    
    mul_hi + (mul_mid >> 32)
}

#[inline]
pub fn butterfly_simd(x: u64x4, t: u64x4, modulus: u64) -> (u64x4, u64x4) {
    let modulus_vec = u64x4::splat(modulus);
    
    // Addition with reduction
    let sum = x + t;
    let sum_reduced = simd_barrett_reduce(sum, modulus_vec);
    
    // Subtraction with modulus normalization
    let modulus_minus_t = modulus_vec - t;
    let diff = x + modulus_minus_t;
    let diff_reduced = simd_barrett_reduce(diff, modulus_vec);
    
    (sum_reduced, diff_reduced)
}

#[inline]
pub fn simd_barrett_reduce(x: u64x4, modulus: u64x4) -> u64x4 {
    // Pre-computed Barrett constant
    let k = 64;
    let r = ((1u128 << k) / modulus[0] as u128) as u64;
    let r_vec = u64x4::splat(r);
    
    let q = (x * r_vec) >> k;
    x - q * modulus
}

pub fn select(mask: Mask<i64, 4>, a: u64x4, b: u64x4) -> u64x4 {
    mask.select(a, b)
}