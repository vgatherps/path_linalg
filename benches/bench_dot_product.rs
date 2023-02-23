use criterion::{black_box, criterion_group, criterion_main, Bencher, Criterion};
use path_linalg::matrix::{MatrixDimensions, MatrixMajority, MatrixMut, OwnedMatrix};
use path_linalg::{dims, matrix::dual_minplus};

fn float_in_neg_1_one() -> f32 {
    (2. * fastrand::f32()) - 1.
}

pub fn criterion_benchmark(c: &mut Criterion) {
    fn do_dot(
        dim_l: MatrixDimensions,
        dim_r: MatrixDimensions,
        dim_o: MatrixDimensions,
        b: &mut Bencher,
    ) {
        let left = OwnedMatrix::<f32>::fill(0., dim_l);
        let right = OwnedMatrix::<f32>::fill(0., dim_r);
        let mut out = OwnedMatrix::<f32>::fill(0., dim_o);
        b.iter(|| {
            let row = black_box(&left);
            let col = black_box(&right);
            let out = black_box(&mut out);
            row.mult::<4>(col, out)
        });
    }

    fn do_minplus(
        dim_l: MatrixDimensions,
        dim_r: MatrixDimensions,
        dim_o: MatrixDimensions,
        b: &mut Bencher,
    ) {
        let left = OwnedMatrix::<f32>::fill(0., dim_l);
        let right = OwnedMatrix::<f32>::fill(0., dim_r);
        let mut out = OwnedMatrix::<f32>::fill(0., dim_o);
        b.iter(|| {
            let row = black_box(&left);
            let col = black_box(&right);
            let out = black_box(&mut out);
            row.minplus::<4>(col, out)
        });
    }

    fn do_dual_minplus(
        dim_l: MatrixDimensions,
        dim_r: MatrixDimensions,
        dim_o: MatrixDimensions,
        b: &mut Bencher,
    ) {
        let mut row_mat = OwnedMatrix::<f32>::fill(0., dim_l);
        let mut row_mat_dual = OwnedMatrix::<f32>::fill(0., dim_l);
        let mut col_mat = OwnedMatrix::<f32>::fill(0., dim_r);
        let mut col_mat_dual = OwnedMatrix::<f32>::fill(0., dim_r);
        let mut out_mat = OwnedMatrix::<f32>::fill(0., dim_o);
        let mut out_mat_dual = OwnedMatrix::<f32>::fill(0., dim_o);

        for f in row_mat
            .data_mut()
            .iter_mut()
            .chain(row_mat_dual.data_mut())
            .chain(col_mat.data_mut().iter_mut())
            .chain(col_mat_dual.data_mut())
            .chain(out_mat.data_mut().iter_mut())
            .chain(out_mat_dual.data_mut())
        {
            *f = float_in_neg_1_one();
        }

        b.iter(|| {
            let row = black_box(&row_mat);
            let col = black_box(&col_mat);
            let out = black_box(&mut out_mat);
            let row_dual = black_box(&row_mat_dual);
            let col_dual = black_box(&col_mat_dual);
            let out_dual = black_box(&mut out_mat_dual);

            dual_minplus::<f32, 4>(row, row_dual, col, col_dual, out, out_dual);
        });
    }

    c.bench_function("64x5x5 dot product rowxcol", |b: &mut Bencher| {
        do_dot(
            dims!(5, 64, MatrixMajority::Row),
            dims!(64, 5, MatrixMajority::Column),
            dims!(5, 5, MatrixMajority::Row),
            b,
        );
    });

    c.bench_function("100x100x1 dot product rowxcol", |b: &mut Bencher| {
        do_dot(
            dims!(100, 100, MatrixMajority::Row),
            dims!(100, 1, MatrixMajority::Column),
            dims!(100, 1, MatrixMajority::Row),
            b,
        );
    });

    c.bench_function("100x100x100 dot product rowxcol", |b: &mut Bencher| {
        do_dot(
            dims!(100, 100, MatrixMajority::Row),
            dims!(100, 100, MatrixMajority::Column),
            dims!(100, 100, MatrixMajority::Row),
            b,
        );
    });

    c.bench_function("64x5x5 minplus product rowxcol", |b: &mut Bencher| {
        do_minplus(
            dims!(5, 64, MatrixMajority::Row),
            dims!(64, 5, MatrixMajority::Column),
            dims!(5, 5, MatrixMajority::Row),
            b,
        );
    });

    c.bench_function("100x100x1 minplus product rowxcol", |b: &mut Bencher| {
        do_minplus(
            dims!(100, 100, MatrixMajority::Row),
            dims!(100, 1, MatrixMajority::Column),
            dims!(100, 1, MatrixMajority::Row),
            b,
        );
    });

    c.bench_function("100x100x100 minplus product rowxcol", |b: &mut Bencher| {
        do_minplus(
            dims!(100, 100, MatrixMajority::Row),
            dims!(100, 100, MatrixMajority::Column),
            dims!(100, 100, MatrixMajority::Row),
            b,
        );
    });

    c.bench_function("64x5x5 dual minplus product rowxcol", |b: &mut Bencher| {
        do_dual_minplus(
            dims!(5, 64, MatrixMajority::Row),
            dims!(64, 5, MatrixMajority::Column),
            dims!(5, 5, MatrixMajority::Row),
            b,
        );
    });

    c.bench_function(
        "100x100x1 dual minplus product rowxcol",
        |b: &mut Bencher| {
            do_dual_minplus(
                dims!(100, 100, MatrixMajority::Row),
                dims!(100, 1, MatrixMajority::Column),
                dims!(100, 1, MatrixMajority::Row),
                b,
            );
        },
    );

    c.bench_function(
        "100x100x100 dual minplus product rowxcol",
        |b: &mut Bencher| {
            do_dual_minplus(
                dims!(100, 100, MatrixMajority::Row),
                dims!(100, 100, MatrixMajority::Column),
                dims!(100, 100, MatrixMajority::Row),
                b,
            );
        },
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
