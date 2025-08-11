use std::marker::PhantomData;

use ecow::EcoVec;

/// Create a new EcoVec, and copy data from the passed in slice into the EcoVec.
///
/// This is kind of hacky, and depends on the ABI of EcoVec<T> remaining stable.
/// EcoVec<T> does have a well defined ABI however, thanks to being `repr(C)`.
pub fn from_slice<T: Copy>(data: &[T]) -> EcoVec<T> {
    #[repr(C)]
    struct TransmutedEcoVec<T> {
        ptr: std::ptr::NonNull<T>,
        len: usize,
        phantom: PhantomData<T>,
    }
    let mut vec = EcoVec::with_capacity(data.len());
    unsafe {
        // Safety: This transmute is fine thanks to repr(C), as long as the EcoVec ABI remains stable.
        let vec: &mut TransmutedEcoVec<T> = std::mem::transmute(&mut vec);
        // Safety:
        // * `src` is valid for reads of `data.len()` bytes.
        // * `dst` is valid for writes of `data.len()` bytes thanks to `EcoVec::with_capacity(data.len())`
        // * `src` and `dst` are properly aligned.
        // * `src` and `dst` are not overlapping since we just allocated `dst`.
        std::ptr::copy_nonoverlapping(data.as_ptr(), vec.ptr.as_ptr(), data.len());
        // Safety: vec capacity is equal to data.len(), and has now been initialized by copy_nonoverlapping
        vec.len = data.len();
    }
    vec
}
