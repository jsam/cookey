use lazy_static::lazy_static;
use std::collections::HashMap;
use std::sync::RwLock;

// Cache structure for storing pre-computed values
#[derive(Debug)]
pub struct RootCache {
    pub roots: HashMap<(u64, u64, usize), u64>,          // (modulus, root, n) -> nth root
    pub inverse_roots: HashMap<(u64, u64, usize), u64>,  // (modulus, root, n) -> inverse nth root
    pub scale_factors: HashMap<(u64, usize), u64>,       // (modulus, n) -> scale factor
}

impl RootCache {
    fn new() -> Self {
        RootCache {
            roots: HashMap::new(),
            inverse_roots: HashMap::new(),
            scale_factors: HashMap::new(),
        }
    }
}

lazy_static! {
    pub static ref ROOT_CACHE: RwLock<RootCache> = RwLock::new(RootCache::new());
}
