use std::{
    ops::{Add, Mul},
    simd::{LaneCount, Mask, Simd, SimdElement, SimdPartialOrd, SupportedLaneCount},
};

use crate::single::accumulators::CanReduceMin;

pub trait DualAccumulator<T1, T2, const LANES: usize> {
    fn init1(&mut self) -> T1;
    fn init2(&mut self) -> T2;

    fn dual_combine(&mut self, acc1: &mut T1, acc2: &mut T2, a1: &T1, a2: &T2, b1: &T1, b2: &T2);

    #[inline]
    fn dual_combine_arr(
        &mut self,
        acc1: &mut [T1; LANES],
        acc2: &mut [T2; LANES],
        a1: &[T1; LANES],
        a2: &[T2; LANES],
        b1: &[T1; LANES],
        b2: &[T2; LANES],
    ) {
        for i in 0..LANES {
            self.dual_combine(&mut acc1[i], &mut acc2[i], &a1[i], &a2[i], &b1[i], &b2[i]);
        }
    }

    fn dual_combine_horizontally(&mut self, acc1: &[T1; LANES], acc2: &[T2; LANES]) -> (T1, T2);
}

pub struct DualMinPlusAcc {}

// TODO this should be various T's
impl<T, const LANES: usize> DualAccumulator<T, T, LANES> for DualMinPlusAcc
where
    T: Mul<Output = T> + Add<Output = T> + SimdElement + Default + PartialOrd + Copy,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: Mul<Output = Simd<T, LANES>>
        + Add<Output = Simd<T, LANES>>
        + CanReduceMin<Output = T>
        + SimdPartialOrd<Mask = Mask<T::Mask, LANES>>,
{
    #[inline]
    fn init1(&mut self) -> T {
        Simd::<T, LANES>::MAX
    }
    #[inline]
    fn init2(&mut self) -> T {
        Simd::<T, LANES>::MAX
    }

    #[inline]
    fn dual_combine(&mut self, acc1: &mut T, acc2: &mut T, a1: &T, a2: &T, b1: &T, b2: &T) {
        let sum1 = *a1 + *b1;
        if sum1 < *acc1 {
            *acc1 = sum1;
            *acc2 = *a2 + *b2;
        }
    }

    #[inline]
    fn dual_combine_arr(
        &mut self,
        acc1: &mut [T; LANES],
        acc2: &mut [T; LANES],
        a1: &[T; LANES],
        a2: &[T; LANES],
        b1: &[T; LANES],
        b2: &[T; LANES],
    ) {
        let mut acc1_simd = Simd::from_array(*acc1);
        let mut acc2_simd = Simd::from_array(*acc2);
        let a1_simd = Simd::from_array(*a1);
        let a2_simd = Simd::from_array(*a2);
        let b1_simd = Simd::from_array(*b1);
        let b2_simd = Simd::from_array(*b2);

        let sum1_simd = a1_simd + b1_simd;
        let mask = sum1_simd.simd_lt(acc1_simd);
        acc1_simd = mask.select(sum1_simd, acc1_simd);
        acc2_simd = mask.select(a2_simd + b2_simd, acc2_simd);
        *acc1 = acc1_simd.to_array();
        *acc2 = acc2_simd.to_array();
    }

    fn dual_combine_horizontally(&mut self, acc1: &[T; LANES], acc2: &[T; LANES]) -> (T, T) {
        let mut val1 = acc1[0];
        let mut val2 = acc2[0];

        for (other, other_dual) in acc1[1..].iter().zip(&acc2[1..]) {
            if *other < val1 {
                val1 = *other;
                val2 = *other_dual;
            }
        }

        (val1, val2)
    }
}
