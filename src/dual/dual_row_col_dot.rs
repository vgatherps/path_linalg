use crate::{
    dual::dual_accumulators::DualAccumulator,
    matrix::{Matrix, MatrixMajority, MatrixMut, MutRefMatrix, RefMatrix},
};

#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn do_accumulate_colmat_block<
    T1: Copy,
    T2: Copy,
    A: DualAccumulator<T1, T2, LANES>,
    const LANES: usize,
    const BLOCK_I: usize,
    const BLOCK_K: usize,
>(
    left1: *const T1,
    left2: *const T2,
    right1: *const T1,
    right2: *const T2,
    into1: *mut T1,
    into2: *mut T2,
    inner_len: usize,
    right_columns: usize,
    accumulator: &mut A,
    i_base: usize,
    k_base: usize,
) {
    let acc_base1 = [accumulator.init1(); LANES];
    let acc_base2 = [accumulator.init2(); LANES];
    let mut acc1 = [[acc_base1; BLOCK_K]; BLOCK_I];
    let mut acc2 = [[acc_base2; BLOCK_K]; BLOCK_I];

    let mut j = 0;
    while (j + LANES) < inner_len {
        for i_off in 0..BLOCK_I {
            for k_off in 0..BLOCK_K {
                let i = i_base + i_off;
                let k = k_base + k_off;
                let row_base1 = left1.add(i * inner_len + j);
                let row_base2 = left2.add(i * inner_len + j);
                let col_base1 = right1.add(k * inner_len + j);
                let col_base2 = right2.add(k * inner_len + j);

                let row_slice1 = &*(row_base1 as *const [T1; LANES]);
                let row_slice2 = &*(row_base2 as *const [T2; LANES]);
                let col_slice1 = &*(col_base1 as *const [T1; LANES]);
                let col_slice2 = &*(col_base2 as *const [T2; LANES]);

                accumulator.dual_combine_arr(
                    &mut acc1[i_off][k_off],
                    &mut acc2[i_off][k_off],
                    row_slice1,
                    row_slice2,
                    col_slice1,
                    col_slice2,
                );
            }
        }

        j += LANES;
    }

    for i_off in 0..BLOCK_I {
        for k_off in 0..BLOCK_K {
            let i = i_base + i_off;
            let k = k_base + k_off;
            let into_idx = i * right_columns + k;
            let write_into1 = &mut *(into1.add(into_idx));
            let write_into2 = &mut *(into2.add(into_idx));
            let (a1, a2) =
                accumulator.dual_combine_horizontally(&acc1[i_off][k_off], &acc2[i_off][k_off]);
            *write_into1 = a1;
            *write_into2 = a2;
        }
    }

    // TODO what's the best order here?
    for j in j..inner_len {
        for i_off in 0..BLOCK_I {
            for k_off in 0..BLOCK_K {
                let i = i_base + i_off;
                let k = k_base + k_off;
                let into_idx = i * right_columns + k;
                let write_to1 = &mut *(into1.add(into_idx));
                let write_to2 = &mut *(into2.add(into_idx));
                let row_base1 = left1.add(i * inner_len + j);
                let row_base2 = left2.add(i * inner_len + j);
                let col_base1 = right1.add(k * inner_len + j);
                let col_base2 = right2.add(k * inner_len + j);

                let row1 = &*row_base1;
                let row2 = &*row_base2;
                let col1 = &*col_base1;
                let col2 = &*col_base2;

                accumulator.dual_combine(write_to1, write_to2, row1, row2, col1, col2);
            }
        }
    }
}

#[inline(always)]
fn do_accumulate_colmat_myrow<
    T1: Copy,
    T2: Copy,
    A: DualAccumulator<T1, T2, LANES>,
    const LANES: usize,
    const BLOCK_I: usize,
>(
    left1: *const T1,
    left2: *const T2,
    right1: *const T1,
    right2: *const T2,
    into1: *mut T1,
    into2: *mut T2,
    inner_len: usize,
    right_columns: usize,
    accumulator: &mut A,
    i: usize,
) {
    unsafe {
        let mut k = 0;
        while (k + 4) <= right_columns && BLOCK_I <= 1 {
            do_accumulate_colmat_block::<T1, T2, A, LANES, BLOCK_I, 4>(
                left1,
                left2,
                right1,
                right2,
                into1,
                into2,
                inner_len,
                right_columns,
                accumulator,
                i,
                k,
            );
            k += 4;
        }
        while (k + 3) <= right_columns && BLOCK_I <= 2 {
            do_accumulate_colmat_block::<T1, T2, A, LANES, BLOCK_I, 3>(
                left1,
                left2,
                right1,
                right2,
                into1,
                into2,
                inner_len,
                right_columns,
                accumulator,
                i,
                k,
            );
            k += 3;
        }
        while (k + 2) <= right_columns && BLOCK_I <= 3 {
            do_accumulate_colmat_block::<T1, T2, A, LANES, BLOCK_I, 2>(
                left1,
                left2,
                right1,
                right2,
                into1,
                into2,
                inner_len,
                right_columns,
                accumulator,
                i,
                k,
            );
            k += 2;
        }
        while (k + 1) <= right_columns {
            do_accumulate_colmat_block::<T1, T2, A, LANES, BLOCK_I, 1>(
                left1,
                left2,
                right1,
                right2,
                into1,
                into2,
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

// This wastes a huge amount of registers just tracking pointers...
// realistitically for optimal speed would just want to force each
// matrix to be 2x size and use the same type?
// Main optimisation is that we want to accumulate path primes in integers
// while accumulating the path costs in fp
#[inline]
pub(crate) fn dual_accumulate_row_col<
    T1: Copy,
    T2: Copy,
    A: DualAccumulator<T1, T2, LANES>,
    const LANES: usize,
>(
    left1: &RefMatrix<T1>,
    left2: &RefMatrix<T2>,
    right1: &RefMatrix<T1>,
    right2: &RefMatrix<T2>,
    into1: &mut MutRefMatrix<T1>,
    into2: &mut MutRefMatrix<T2>,
    mut accumulator: A,
) {
    assert_eq!(left1.dimensions(), left2.dimensions());
    assert_eq!(right1.dimensions(), right2.dimensions());
    assert_eq!(into1.dimensions(), into2.dimensions());
    assert_eq!(left1.majority(), MatrixMajority::Row);
    assert_eq!(right1.majority(), MatrixMajority::Column);
    assert_eq!(right1.rows(), left1.columns());
    assert_eq!(left1.rows(), into1.rows());
    assert_eq!(right1.columns(), into1.columns());

    let inner_len = left1.columns().get();
    let left_rows = left1.rows().get();
    let right_columns = right1.columns().get();

    let left1 = left1.data().as_ptr();
    let left2 = left2.data().as_ptr();
    let right1 = right1.data().as_ptr();
    let right2 = right2.data().as_ptr();
    let into1 = into1.data_mut().as_mut_ptr();
    let into2 = into2.data_mut().as_mut_ptr();

    let mut i = 0;
    while (i + 4) <= left_rows {
        do_accumulate_colmat_myrow::<T1, T2, A, LANES, 4>(
            left1,
            left2,
            right1,
            right2,
            into1,
            into2,
            inner_len,
            right_columns,
            &mut accumulator,
            i,
        );
        i += 4;
    }
    while (i + 3) <= left_rows {
        do_accumulate_colmat_myrow::<T1, T2, A, LANES, 3>(
            left1,
            left2,
            right1,
            right2,
            into1,
            into2,
            inner_len,
            right_columns,
            &mut accumulator,
            i,
        );
        i += 3;
    }
    while (i + 2) <= left_rows {
        do_accumulate_colmat_myrow::<T1, T2, A, LANES, 2>(
            left1,
            left2,
            right1,
            right2,
            into1,
            into2,
            inner_len,
            right_columns,
            &mut accumulator,
            i,
        );
        i += 2;
    }
    while (i + 1) <= left_rows {
        do_accumulate_colmat_myrow::<T1, T2, A, LANES, 1>(
            left1,
            left2,
            right1,
            right2,
            into1,
            into2,
            inner_len,
            right_columns,
            &mut accumulator,
            i,
        );
        i += 1;
    }
}
