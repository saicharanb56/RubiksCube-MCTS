use criterion::{black_box, criterion_group, criterion_main, Criterion};

use rubikscube::{Cube, MetricKind};

fn criterion_benchmark(c: &mut Criterion) {
    let mut cube = Cube::new(MetricKind::HalfTurnMetric);
    c.bench_function("cube turn", |b| b.iter(|| cube.turn(black_box(0))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
