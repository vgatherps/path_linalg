use std::{
    num::NonZeroUsize,
    ops::{Add, AddAssign, Mul},
    simd::{LaneCount, Mask, Simd, SimdElement, SimdPartialOrd, SupportedLaneCount},
};

use crate::{
    dual::{dual_accumulators::DualMinPlusAcc, dual_row_col_dot::dual_accumulate_row_col},
    single::{
        accumulators::{Accumulator, CanReduceAdd, CanReduceMin, MinPlusAcc, MultAcc},
        row_col_dot::accumulate_row_col,
    },
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MatrixMajority {
    Row,
    Column,
}

impl MatrixMajority {
    #[inline]
    pub fn index(&self, row: usize, rows: usize, column: usize, columns: usize) -> usize {
        match self {
            MatrixMajority::Row => (row * columns) + column,
            MatrixMajority::Column => (column * rows) + row,
        }
    }

    #[inline]
    pub fn other(&self) -> MatrixMajority {
        match self {
            MatrixMajority::Column => MatrixMajority::Row,
            MatrixMajority::Row => MatrixMajority::Column,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MatrixDimensions {
    pub rows: NonZeroUsize,
    pub columns: NonZeroUsize,
    pub majority: MatrixMajority,
}

impl MatrixDimensions {
    #[inline]
    pub fn new(rows: usize, columns: usize, majority: MatrixMajority) -> Self {
        Self {
            rows: NonZeroUsize::new(rows).unwrap(),
            columns: NonZeroUsize::new(columns).unwrap(),
            majority,
        }
    }
}

#[macro_export]
macro_rules! dims {
    ($rows:expr, $columns:expr) => {
        $crate::matrix::MatrixDimensions::new($rows, $columns, $crate::matrix::MatrixMajority::Row)
    };
    ($rows:expr, $columns:expr, $majority:expr) => {
        $crate::matrix::MatrixDimensions::new($rows, $columns, $majority)
    };
}

pub trait Matrix<T> {
    type Storage<'a>: 'a
    where
        Self: 'a;
    fn dimensions_mut(&mut self) -> &mut MatrixDimensions;
    fn dimensions(&self) -> MatrixDimensions;
    fn data(&self) -> &[T];
    #[inline]
    fn rows(&self) -> NonZeroUsize {
        self.dimensions().rows
    }
    #[inline]
    fn columns(&self) -> NonZeroUsize {
        self.dimensions().columns
    }
    #[inline]
    fn majority(&self) -> MatrixMajority {
        self.dimensions().majority
    }
    #[inline]
    fn to_ref(&self) -> RefMatrix<T> {
        RefMatrix {
            dimensions: self.dimensions(),
            data: self.data(),
        }
    }
    #[inline]
    fn at(&self, row: usize, column: usize) -> &T {
        &self.data()[self
            .majority()
            .index(row, self.rows().get(), column, self.columns().get())]
    }
}

pub trait MatrixMut<T>: Matrix<T> {
    fn data_mut(&mut self) -> &mut [T];

    #[inline]
    fn to_mut_ref(&mut self) -> MutRefMatrix<T> {
        MutRefMatrix {
            dimensions: self.dimensions(),
            data: self.data_mut(),
        }
    }
    #[inline]
    fn at_mut(&mut self, row: usize, column: usize) -> &mut T {
        let idx = self
            .majority()
            .index(row, self.rows().get(), column, self.columns().get());
        &mut self.data_mut()[idx]
    }
}

#[derive(Clone, Debug)]
pub struct OwnedMatrix<T> {
    dimensions: MatrixDimensions,
    data: Vec<T>,
}

#[derive(Copy, Clone, Debug)]
pub struct RefMatrix<'a, T> {
    dimensions: MatrixDimensions,
    data: &'a [T],
}

#[derive(Debug)]
pub struct MutRefMatrix<'a, T> {
    dimensions: MatrixDimensions,
    data: &'a mut [T],
}

impl<T> Matrix<T> for OwnedMatrix<T> {
    type Storage<'b> = Vec<T> where Self: 'b;

    #[inline]
    fn dimensions(&self) -> MatrixDimensions {
        self.dimensions
    }

    #[inline]
    fn dimensions_mut(&mut self) -> &mut MatrixDimensions {
        &mut self.dimensions
    }

    #[inline]
    fn data(&self) -> &[T] {
        &self.data
    }
}

impl<'a, T: 'a> Matrix<T> for RefMatrix<'a, T> {
    type Storage<'b> = &'a mut [T] where 'a: 'b;

    #[inline]
    fn dimensions(&self) -> MatrixDimensions {
        self.dimensions
    }

    #[inline]
    fn dimensions_mut(&mut self) -> &mut MatrixDimensions {
        &mut self.dimensions
    }

    #[inline]
    fn data(&self) -> &[T] {
        self.data
    }
}

impl<'a, T: 'a> Matrix<T> for MutRefMatrix<'a, T> {
    type Storage<'b> = &'a mut [T] where 'a: 'b;

    #[inline]
    fn dimensions(&self) -> MatrixDimensions {
        self.dimensions
    }

    #[inline]
    fn dimensions_mut(&mut self) -> &mut MatrixDimensions {
        &mut self.dimensions
    }

    #[inline]
    fn data(&self) -> &[T] {
        self.data
    }
}

impl<T> MatrixMut<T> for OwnedMatrix<T> {
    #[inline]
    fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<'a, T: 'a> MatrixMut<T> for MutRefMatrix<'a, T> {
    #[inline]
    fn data_mut(&mut self) -> &mut [T] {
        self.data
    }
}

// TODO is this actually any better/worse than the same but with a fully dynamic matrix?
impl<T: Copy> OwnedMatrix<T> {
    pub fn fill(val: T, dimensions: MatrixDimensions) -> OwnedMatrix<T> {
        let total_length = dimensions.rows.get() * dimensions.columns.get();
        Self {
            data: vec![val; total_length],
            dimensions,
        }
    }

    #[inline]
    pub fn run_acc<A: Accumulator<T, LANES>, const LANES: usize>(
        &self,
        other: &OwnedMatrix<T>,
        into: &mut OwnedMatrix<T>,
        acc: A,
    ) {
        if self.majority() == MatrixMajority::Row && other.majority() == MatrixMajority::Column {
            into.dimensions.majority = MatrixMajority::Row;
            accumulate_row_col::<T, A, LANES>(
                &self.to_ref(),
                &other.to_ref(),
                &mut into.to_mut_ref(),
                acc,
            )
        } else {
            panic!("Must implement the other rows");
        }
    }

    pub fn copy_from(&mut self, other: &OwnedMatrix<T>) {
        assert_eq!(self.rows(), other.rows());
        assert_eq!(self.columns(), other.columns());
        self.dimensions = other.dimensions;
        self.data.copy_from_slice(&other.data);
    }
}

impl<T> OwnedMatrix<T> {
    #[inline]
    pub fn mult<const LANES: usize>(&self, other: &OwnedMatrix<T>, into: &mut OwnedMatrix<T>)
    where
        T: Mul<Output = T> + AddAssign + SimdElement + Default,
        LaneCount<LANES>: SupportedLaneCount,
        Simd<T, LANES>:
            Mul<Output = Simd<T, LANES>> + Add<Output = Simd<T, LANES>> + CanReduceAdd<Output = T>,
    {
        let mult_acc = MultAcc {};
        self.run_acc::<MultAcc, LANES>(other, into, mult_acc)
    }

    #[inline]
    pub fn minplus<const LANES: usize>(&self, other: &OwnedMatrix<T>, into: &mut OwnedMatrix<T>)
    where
        T: Mul<Output = T> + Add<Output = T> + SimdElement + Default + PartialOrd + Copy,
        LaneCount<LANES>: SupportedLaneCount,
        Simd<T, LANES>:
            Mul<Output = Simd<T, LANES>> + Add<Output = Simd<T, LANES>> + CanReduceMin<Output = T>,
    {
        let mult_acc = MinPlusAcc {};
        self.run_acc::<MinPlusAcc, LANES>(other, into, mult_acc)
    }

    #[inline]
    pub fn lazy_transpose(self) -> OwnedMatrix<T> {
        Self {
            dimensions: dims!(
                self.rows().get(),
                self.columns().get(),
                self.majority().other()
            ),
            data: self.data,
        }
    }
}

pub fn dual_minplus<T, const LANES: usize>(
    a: &OwnedMatrix<T>,
    a_dual: &OwnedMatrix<T>,
    b: &OwnedMatrix<T>,
    b_dual: &OwnedMatrix<T>,
    into: &mut OwnedMatrix<T>,
    into_dual: &mut OwnedMatrix<T>,
) where
    T: Mul<Output = T> + Add<Output = T> + SimdElement + Default + PartialOrd + Copy,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: Mul<Output = Simd<T, LANES>>
        + Add<Output = Simd<T, LANES>>
        + CanReduceMin<Output = T>
        + SimdPartialOrd<Mask = Mask<T::Mask, LANES>>,
{
    assert!(a.dimensions() == a_dual.dimensions());
    assert!(b.dimensions() == b_dual.dimensions());
    assert!(into.dimensions() == into_dual.dimensions());

    if a.majority() == MatrixMajority::Row && b.majority() == MatrixMajority::Column {
        into.dimensions.majority = MatrixMajority::Row;
        into_dual.dimensions.majority = MatrixMajority::Row;
        dual_accumulate_row_col::<T, T, DualMinPlusAcc, LANES>(
            &a.to_ref(),
            &a_dual.to_ref(),
            &b.to_ref(),
            &b_dual.to_ref(),
            &mut into.to_mut_ref(),
            &mut into_dual.to_mut_ref(),
            DualMinPlusAcc {},
        )
    }
}
