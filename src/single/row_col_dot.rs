use std::mem::MaybeUninit;

use crate::{
    matrix::{Matrix, MatrixMajority, MatrixMut, MutRefMatrix, RefMatrix},
    single::accumulators::Accumulator,
};

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn do_accumulate_colmat_block<
    T: Copy,
    A: Accumulator<T, LANES>,
    const LANES: usize,
    const BLOCK_I: usize,
    const BLOCK_K: usize,
>(
    left: *const T,
    right: *const T,
    into: *mut T,
    inner_len: usize,
    right_columns: usize,
    accumulator: &mut A,
    i_base: usize,
    k_base: usize,
) {
    let acc_base = [accumulator.init(); LANES];
    let mut acc = [[acc_base; BLOCK_K]; BLOCK_I];

    let mut j = 0;
    while (j + LANES) < inner_len {
        for i_off in 0..BLOCK_I {
            for k_off in 0..BLOCK_K {
                let i = i_base + i_off;
                let k = k_base + k_off;
                let row_base = left.add(i * inner_len + j);
                let col_base = right.add(k * inner_len + j);

                let row_slice = &*(row_base as *const [T; LANES]);
                let col_slice = &*(col_base as *const [T; LANES]);

                accumulator.combine_arr(&mut acc[i_off][k_off], row_slice, col_slice);
            }
        }

        j += LANES;
    }

    // the array map methods are terrifyingly bad, codegen is awful...
    let mut combined = MaybeUninit::<[[T; BLOCK_K]; BLOCK_I]>::uninit();
    let write_into = &mut *combined.as_mut_ptr();

    for i in 0..BLOCK_I {
        for k in 0..BLOCK_K {
            write_into[i][k] = accumulator.combine_horizontally(&acc[i][k]);
        }
    }

    let mut acc = combined.assume_init();

    for j in j..inner_len {
        for i_off in 0..BLOCK_I {
            for k_off in 0..BLOCK_K {
                let i = i_base + i_off;
                let k = k_base + k_off;
                let row_base = left.add(i * inner_len + j);
                let col_base = right.add(k * inner_len + j);

                let row = &*row_base;
                let col = &*col_base;

                accumulator.combine(&mut acc[i_off][k_off], row, col);
            }
        }
    }

    for i_off in 0..BLOCK_I {
        for k_off in 0..BLOCK_K {
            let i = i_base + i_off;
            let k = k_base + k_off;
            let into_idx = i * right_columns + k;
            let write_to = &mut *(into.add(into_idx));
            *write_to = acc[i_off][k_off];
        }
    }
}

#[inline(always)]
fn do_accumulate_colmat_myrow<
    T: Copy,
    A: Accumulator<T, LANES>,
    const LANES: usize,
    const BLOCK_I: usize,
>(
    left: *const T,
    right: *const T,
    into: *mut T,
    inner_len: usize,
    right_columns: usize,
    accumulator: &mut A,
    i: usize,
) {
    unsafe {
        let mut k = 0;
        while (k + 4) <= right_columns && BLOCK_I <= 1 {
            do_accumulate_colmat_block::<T, A, LANES, BLOCK_I, 4>(
                left,
                right,
                into,
                inner_len,
                right_columns,
                accumulator,
                i,
                k,
            );
            k += 4;
        }
        while (k + 3) <= right_columns && BLOCK_I <= 2 {
            do_accumulate_colmat_block::<T, A, LANES, BLOCK_I, 3>(
                left,
                right,
                into,
                inner_len,
                right_columns,
                accumulator,
                i,
                k,
            );
            k += 3;
        }
        while (k + 2) <= right_columns && BLOCK_I <= 3 {
            do_accumulate_colmat_block::<T, A, LANES, BLOCK_I, 2>(
                left,
                right,
                into,
                inner_len,
                right_columns,
                accumulator,
                i,
                k,
            );
            k += 2;
        }
        while (k + 1) <= right_columns {
            do_accumulate_colmat_block::<T, A, LANES, BLOCK_I, 1>(
                left,
                right,
                into,
                inner_len,
                right_columns,
                accumulator,
                i,
                k,
            );
            k += 1;
        }
    }
}

#[inline]
pub(crate) fn accumulate_row_col<T: Copy, A: Accumulator<T, LANES>, const LANES: usize>(
    left: &RefMatrix<T>,
    right: &RefMatrix<T>,
    into: &mut MutRefMatrix<T>,
    mut accumulator: A,
) {
    assert_eq!(left.majority(), MatrixMajority::Row);
    assert_eq!(right.majority(), MatrixMajority::Column);
    assert_eq!(right.rows(), left.columns());
    assert_eq!(left.rows(), into.rows());
    assert_eq!(right.columns(), into.columns());

    let inner_len = left.columns().get();
    let left_rows = left.rows().get();
    let right_columns = right.columns().get();

    let left = left.data().as_ptr();
    let right = right.data().as_ptr();
    let into = into.data_mut().as_mut_ptr();

    let mut i = 0;
    while (i + 4) <= left_rows {
        do_accumulate_colmat_myrow::<T, A, LANES, 4>(
            left,
            right,
            into,
            inner_len,
            right_columns,
            &mut accumulator,
            i,
        );
        i += 4;
    }
    while (i + 3) <= left_rows {
        do_accumulate_colmat_myrow::<T, A, LANES, 3>(
            left,
            right,
            into,
            inner_len,
            right_columns,
            &mut accumulator,
            i,
        );
        i += 3;
    }
    while (i + 2) <= left_rows {
        do_accumulate_colmat_myrow::<T, A, LANES, 2>(
            left,
            right,
            into,
            inner_len,
            right_columns,
            &mut accumulator,
            i,
        );
        i += 2;
    }
    while (i + 1) <= left_rows {
        do_accumulate_colmat_myrow::<T, A, LANES, 1>(
            left,
            right,
            into,
            inner_len,
            right_columns,
            &mut accumulator,
            i,
        );
        i += 1;
    }
}
