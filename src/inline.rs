// Helper function to compute log2 of a number
#[inline]
pub fn log2(n: usize) -> u32 {
    n.trailing_zeros()
}

// Helper function to reverse bits of a number with specific width
#[inline]
pub fn reverse_bits(num: usize, width: u32) -> usize {
    let mut result = 0;
    for i in 0..width {
        if (num & (1 << i)) != 0 {
            result |= 1 << (width - 1 - i);
        }
    }
    result
}

// Helper function to reverse bits of a number with specific width in parallel.
/* 
NOTE: No, passing by reference isn't needed here for performance. `usize`` is a primitive type that implements Copy by default, 
and it's typically just a 64-bit or 32-bit integer (depending on platform). 
Copying a usize is just as efficient as copying a reference to it - both are single machine words.
In fact, passing a reference would be slightly less efficient because:

We'd need to dereference the pointer to get the value
The value needs to be copied into registers for bitwise operations anyway
A reference takes up the same space as the usize itself
*/
#[inline]
pub fn reverse_bits_par(x: usize, width: u32) -> usize {
    let mut x = x;
    x = ((x & 0x5555555555555555) << 1) | ((x & 0xAAAAAAAAAAAAAAAA) >> 1);
    x = ((x & 0x3333333333333333) << 2) | ((x & 0xCCCCCCCCCCCCCCCC) >> 2);
    x = ((x & 0x0F0F0F0F0F0F0F0F) << 4) | ((x & 0xF0F0F0F0F0F0F0F0) >> 4);
    x = ((x & 0x00FF00FF00FF00FF) << 8) | ((x & 0xFF00FF00FF00FF00) >> 8);
    x = ((x & 0x0000FFFF0000FFFF) << 16) | ((x & 0xFFFF0000FFFF0000) >> 16);
    x = ((x & 0x00000000FFFFFFFF) << 32) | ((x & 0xFFFFFFFF00000000) >> 32);
    x >> (64 - width)
}


// Helper function to get prime factors of n
#[inline]
pub fn prime_factors(mut n: u64) -> Vec<u64> {
    let mut factors = Vec::new();
    let mut d = 2;
    
    while n > 1 {
        while n % d == 0 {
            if factors.last() != Some(&d) {
                factors.push(d);
            }
            n /= d;
        }
        d += if d == 2 { 1 } else { 2 };
        if d * d > n {
            if n > 1 {
                factors.push(n);
            }
            break;
        }
    }
    
    factors
}

// Const table generation for compile-time computation
#[inline]
pub const fn generate_bit_reverse_table_8() -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        let mut bits = 0u8;
        let mut j = 0;
        while j < 8 {
            if (i & (1 << j)) != 0 {
                bits |= 1 << (7 - j);
            }
            j += 1;
        }
        table[i as usize] = bits;
        i += 1;
    }
    table
}


// bit reverse lookup table for small input size (n <= 256)
const BIT_REVERSE_TABLE_8: [u8; 256] = generate_bit_reverse_table_8();


// Fast bit reversal for different sizes
#[inline(always)]
pub fn reverse_bits_fast(num: usize, width: u32) -> usize {
    match width {
        8 => BIT_REVERSE_TABLE_8[num as u8 as usize] as usize,
        16 => {
            let low = BIT_REVERSE_TABLE_8[(num & 0xFF) as usize] as usize;
            let high = BIT_REVERSE_TABLE_8[((num >> 8) & 0xFF) as usize] as usize;
            (low << 8) | high
        }
        32 => {
            let b0 = BIT_REVERSE_TABLE_8[(num & 0xFF) as usize] as usize;
            let b1 = BIT_REVERSE_TABLE_8[((num >> 8) & 0xFF) as usize] as usize;
            let b2 = BIT_REVERSE_TABLE_8[((num >> 16) & 0xFF) as usize] as usize;
            let b3 = BIT_REVERSE_TABLE_8[((num >> 24) & 0xFF) as usize] as usize;
            (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
        }
        _ => {
            // Fallback to runtime computation for larger widths
            let mut result = 0;
            for i in 0..width {
                if (num & (1 << i)) != 0 {
                    result |= 1 << (width - 1 - i);
                }
            }
            result
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_reverse_table_8() {
        for i in 0..256 {
            let expected = reverse_bits(i, 8);
            let actual = BIT_REVERSE_TABLE_8[i] as usize;
            assert_eq!(actual, expected, "Failed for input {}", i);
        }
    }

    #[test]
    fn test_bit_reverse_fast() {
        // Test 8-bit reversal
        for i in 0..256 {
            let expected = reverse_bits(i, 8);
            let actual = reverse_bits_fast(i, 8);
            assert_eq!(actual, expected, "Failed for 8-bit input {}", i);
        }

        // Test 16-bit reversal
        for i in 0..1000 {
            let expected = reverse_bits(i, 16);
            let actual = reverse_bits_fast(i, 16);
            assert_eq!(actual, expected, "Failed for 16-bit input {}", i);
        }
    }
}
