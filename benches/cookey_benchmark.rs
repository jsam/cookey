use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use cookey::fields::finite::FiniteField;
use cookey::traits::Transform;


const PRIME: u64 = 0x1fffffffffe00001;
const ROOT: u64 = 0x15eb043c7aa2b01f;
const ROOT_ORDER: u64 = 1 << 17;  // 2^17
const MAX_N: usize = 1 << 16;     // 2^16

fn bench_ntt_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt_transform");
    group.sample_size(100); // Reduce sample size for large transforms
    
    let field = FiniteField::new(PRIME, ROOT, ROOT_ORDER);
    
    // Benchmark powers of 2 up to MAX_N
    for &size in &[1024, 2048, 4096, 8192, 16384, MAX_N] {
        // Generate test data
        let data = field.inputs()
            .extend((0..size).map(|x| x as u64))
            .build();
        
        let mut _forward = data.clone();
        // Forward NTT
        group.bench_with_input(
            BenchmarkId::new("forward", size), 
            &size,
            |b, _| b.iter(|| {
                field.transform_forward(black_box(&mut _forward));
            })
        );
        
        let mut _inverse = _forward.clone();
        // Inverse NTT
        group.bench_with_input(
            BenchmarkId::new("inverse", size), 
            &size,
            |b, _| b.iter(|| {
                field.transform_inverse(black_box(&mut _inverse));
            })
        );
    }
    group.finish();
}


fn bench_ntt_convolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt_convolution");
    group.sample_size(20);
    
    let field = FiniteField::new(PRIME, ROOT, ROOT_ORDER);
    
    for &size in &[1024, 2048, 4096, 8192, 16384, MAX_N] {
        let mut v1 = field.inputs()
            .extend((0..size).map(|x| x as u64))
            .build();
        let mut v2 = field.inputs()
            .extend((0..size).map(|x| (x * 2) as u64))
            .build();

        group.bench_with_input(
            BenchmarkId::new("convolution", size),
            &size,
            |b, _| b.iter(|| {
                field.transform_forward(black_box(&mut v1));
                field.transform_forward(black_box(&mut v2));
                
                let mut conv: Vec<cookey::fields::finite::FiniteFieldElement<'_>> = v1.iter().zip(v2.iter())
                    .map(|(x, y)| x.clone() * y.clone())
                    .collect::<Vec<_>>();
                
                field.transform_inverse(black_box(&mut conv));
            })
        );
    }
    group.finish();
}

criterion_group!(benches, bench_ntt_transform, bench_ntt_convolution);
criterion_main!(benches);