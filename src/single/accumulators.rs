use std::{
    ops::{Add, AddAssign, Mul},
    simd::{LaneCount, Simd, SimdElement, SimdFloat, SimdOrd, SimdUint, SupportedLaneCount},
};

pub trait Accumulator<T, const LANES: usize> {
    fn init(&mut self) -> T;

    fn combine(&mut self, acc: &mut T, a: &T, b: &T);

    #[inline]
    fn combine_arr(&mut self, acc: &mut [T; LANES], a: &[T; LANES], b: &[T; LANES]) {
        for i in 0..LANES {
            self.combine(&mut acc[i], &a[i], &b[i]);
        }
    }

    fn combine_horizontally(&mut self, acc: &[T; LANES]) -> T;
}

pub struct MultAcc {}
pub struct MinPlusAcc {}

impl<T, const LANES: usize> Accumulator<T, LANES> for MultAcc
where
    T: Mul<Output = T> + AddAssign + SimdElement + Default,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>:
        Mul<Output = Simd<T, LANES>> + Add<Output = Simd<T, LANES>> + CanReduceAdd<Output = T>,
{
    #[inline]
    fn init(&mut self) -> T {
        T::default()
    }

    #[inline]
    fn combine(&mut self, acc: &mut T, a: &T, b: &T) {
        *acc += *a * *b;
    }

    #[inline]
    fn combine_arr(&mut self, acc: &mut [T; LANES], a: &[T; LANES], b: &[T; LANES]) {
        let mut acc_simd = Simd::from_array(*acc);
        acc_simd = Simd::from_array(*a) * Simd::from_array(*b) + acc_simd;
        *acc = acc_simd.to_array();
    }

    #[inline]
    fn combine_horizontally(&mut self, acc: &[T; LANES]) -> T {
        Simd::from_array(*acc).reduce_add()
    }
}

impl<T, const LANES: usize> Accumulator<T, LANES> for MinPlusAcc
where
    T: Mul<Output = T> + Add<Output = T> + SimdElement + Default + PartialOrd + Copy,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>:
        Mul<Output = Simd<T, LANES>> + Add<Output = Simd<T, LANES>> + CanReduceMin<Output = T>,
{
    #[inline]
    fn init(&mut self) -> T {
        Simd::<T, LANES>::MAX
    }

    #[inline]
    fn combine(&mut self, acc: &mut T, a: &T, b: &T) {
        let sum = *a + *b;
        if sum < *acc {
            *acc = sum;
        }
    }

    #[inline]
    fn combine_arr(&mut self, acc: &mut [T; LANES], a: &[T; LANES], b: &[T; LANES]) {
        let mut acc_simd = Simd::from_array(*acc);
        acc_simd = acc_simd.min_of(Simd::from_array(*a) + Simd::from_array(*b));
        *acc = acc_simd.to_array();
    }

    #[inline]
    fn combine_horizontally(&mut self, acc: &[T; LANES]) -> T {
        Simd::from_array(*acc).reduce_minimum()
    }
}

pub trait CanReduceAdd {
    type Output;
    fn reduce_add(self) -> Self::Output;
}

impl<const L: usize> CanReduceAdd for Simd<f32, L>
where
    LaneCount<L>: SupportedLaneCount,
{
    type Output = f32;
    #[inline]
    fn reduce_add(self) -> Self::Output {
        self.reduce_sum()
    }
}

impl<const L: usize> CanReduceAdd for Simd<f64, L>
where
    LaneCount<L>: SupportedLaneCount,
{
    type Output = f64;
    #[inline]
    fn reduce_add(self) -> Self::Output {
        self.reduce_sum()
    }
}

impl<const L: usize> CanReduceAdd for Simd<u32, L>
where
    LaneCount<L>: SupportedLaneCount,
{
    type Output = u32;
    #[inline]
    fn reduce_add(self) -> Self::Output {
        self.reduce_sum()
    }
}

impl<const L: usize> CanReduceAdd for Simd<u64, L>
where
    LaneCount<L>: SupportedLaneCount,
{
    type Output = u64;
    #[inline]
    fn reduce_add(self) -> Self::Output {
        self.reduce_sum()
    }
}

pub trait CanReduceMin {
    type Output;
    const MAX: Self::Output;
    fn reduce_minimum(self) -> Self::Output;
    fn min_of(self, other: Self) -> Self;
}

impl<const L: usize> CanReduceMin for Simd<f32, L>
where
    LaneCount<L>: SupportedLaneCount,
{
    type Output = f32;
    const MAX: Self::Output = std::f32::INFINITY;
    #[inline]
    fn reduce_minimum(self) -> Self::Output {
        self.reduce_min()
    }

    #[inline]
    fn min_of(self, other: Self) -> Self {
        self.simd_min(other)
    }
}

impl<const L: usize> CanReduceMin for Simd<f64, L>
where
    LaneCount<L>: SupportedLaneCount,
{
    type Output = f64;
    const MAX: Self::Output = f64::INFINITY;
    #[inline]
    fn reduce_minimum(self) -> Self::Output {
        self.reduce_min()
    }

    #[inline]
    fn min_of(self, other: Self) -> Self {
        self.simd_min(other)
    }
}

impl<const L: usize> CanReduceMin for Simd<u32, L>
where
    LaneCount<L>: SupportedLaneCount,
{
    type Output = u32;
    const MAX: Self::Output = u32::MAX;
    #[inline]
    fn reduce_minimum(self) -> Self::Output {
        self.reduce_min()
    }
    #[inline]
    fn min_of(self, other: Self) -> Self {
        self.simd_min(other)
    }
}

impl<const L: usize> CanReduceMin for Simd<u64, L>
where
    LaneCount<L>: SupportedLaneCount,
{
    type Output = u64;
    const MAX: Self::Output = u64::MAX;
    #[inline]
    fn reduce_minimum(self) -> Self::Output {
        self.reduce_min()
    }
    #[inline]
    fn min_of(self, other: Self) -> Self {
        self.simd_min(other)
    }
}
