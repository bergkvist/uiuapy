use ecow::EcoVec;
use numpy::npyffi::{NPY_TYPES, NpyTypes, PyArrayObject, PyDataType_ELSIZE, PyDataType_SET_ELSIZE};
use numpy::{PY_ARRAY_API, PyUntypedArray, npyffi};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyValueError;
use pyo3::ffi::Py_IS_TYPE;
use pyo3::prelude::*;
use std::marker::PhantomData;
use uiua::{Boxed, Complex, Value};

pub fn timed<T>(label: &str, f: impl FnOnce() -> T) -> T {
    let t0 = std::time::Instant::now();
    let result = f();
    println!("{label}: {:?}", t0.elapsed());
    result
}

pub fn numpy_array_to_uiua_value<'py>(py: Python<'py>, array: &Py<PyAny>) -> PyResult<Value> {
    unsafe {
        if Py_IS_TYPE(
            array.as_ptr(),
            PY_ARRAY_API.get_type_object(py, NpyTypes::PyArray_Type),
        ) == 0
        {
            return Err(PyValueError::new_err(format!(
                "Not a numpy array: {array:?}"
            )));
        }
        let arr = array.as_ptr() as *mut PyArrayObject;
        let shape = pyarray_dims(arr);
        let ty: NPY_TYPES = std::mem::transmute((*(*arr).descr).type_num);
        match ty {
            NPY_TYPES::NPY_UBYTE => {
                let data = eco_vec_from_numpy_array(arr, &shape);
                Ok(Value::Byte(uiua::Array::<u8>::new(shape, data)))
            }
            NPY_TYPES::NPY_DOUBLE => {
                let data = eco_vec_from_numpy_array(arr, &shape);
                Ok(Value::Num(uiua::Array::<f64>::new(shape, data)))
            }
            NPY_TYPES::NPY_CDOUBLE => {
                let data = eco_vec_from_numpy_array(arr, &shape);
                Ok(Value::Complex(uiua::Array::<Complex>::new(shape, data)))
            }
            NPY_TYPES::NPY_UNICODE => {
                let mut shape = shape;
                shape.push(PyDataType_ELSIZE(py, (*arr).descr) as usize >> 2);
                let data = eco_vec_from_numpy_array(arr, &shape);
                Ok(Value::Char(uiua::Array::<char>::new(shape, data)))
            }
            NPY_TYPES::NPY_OBJECT => {
                let slice = pyarray_data::<Py<PyAny>>(arr, &shape);
                let values = slice
                    .iter()
                    .map(|x| numpy_array_to_uiua_value(py, x).map(Boxed))
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(Value::Box(uiua::Array::<Boxed>::new(
                    shape,
                    values.as_slice(),
                )))
            }
            NPY_TYPES::NPY_BOOL => {
                let slice = pyarray_data::<bool>(arr, &shape);
                let slice = slice.iter().copied().map(|x| x as u8).collect::<Vec<_>>();
                Ok(Value::Byte(uiua::Array::<u8>::new(shape, slice.as_slice())))
            }
            NPY_TYPES::NPY_LONG => {
                let slice = pyarray_data::<i64>(arr, &shape);
                let slice = slice.iter().copied().map(|x| x as f64).collect::<Vec<_>>();
                Ok(Value::Num(uiua::Array::<f64>::new(shape, slice.as_slice())))
            }
            NPY_TYPES::NPY_ULONG => {
                let slice = pyarray_data::<u64>(arr, &shape);
                let slice = slice.iter().copied().map(|x| x as f64).collect::<Vec<_>>();
                Ok(Value::Num(uiua::Array::<f64>::new(shape, slice.as_slice())))
            }
            _ => Err(PyValueError::new_err(format!(
                "Unsupported numpy array type: {ty:?}"
            ))),
        }
    }
}

unsafe fn pyarray_data<'a, T>(arr: *mut PyArrayObject, dims: &[usize]) -> &'a [T] {
    let len = dims.iter().copied().product::<usize>();
    unsafe {
        let data = (*arr).data.cast::<T>();
        std::ptr::slice_from_raw_parts(data, len).as_ref().unwrap()
    }
}

unsafe fn pyarray_dims(arr: *mut PyArrayObject) -> Vec<usize> {
    unsafe {
        (0..(*arr).nd)
            .map(|i| (*arr).dimensions.add(i as usize).read() as usize)
            .collect()
    }
}

pub fn uiua_value_to_numpy_array<'py>(py: Python<'py>, value: Value) -> PyResult<Py<PyAny>> {
    let mut dims = value.shape.iter().copied().collect::<Vec<_>>();
    match value {
        Value::Num(values) => {
            let data = values.elements().copied().collect::<Vec<_>>();
            pyarray_new_from_data(py, &dims, NPY_TYPES::NPY_DOUBLE, None, &data)
        }
        Value::Byte(values) => {
            let data = values.elements().copied().collect::<Vec<_>>();
            pyarray_new_from_data(py, &dims, NPY_TYPES::NPY_UBYTE, None, &data)
        }
        Value::Complex(values) => {
            let data = values.elements().copied().collect::<Vec<_>>();
            pyarray_new_from_data(py, &dims, NPY_TYPES::NPY_CDOUBLE, None, &data)
        }
        Value::Char(values) => {
            let data = values.elements().copied().collect::<Vec<_>>();
            let elem_size = Some(4 * dims.pop().unwrap_or(0));
            pyarray_new_from_data(py, &dims, NPY_TYPES::NPY_UNICODE, elem_size, &data)
        }
        Value::Box(values) => {
            let data = values
                .elements()
                .map(|Boxed(value)| uiua_value_to_numpy_array(py, value.clone()))
                .collect::<PyResult<Vec<_>>>()?;
            pyarray_new_from_data(py, &dims, NPY_TYPES::NPY_OBJECT, None, &data)
        }
    }
}

pub fn pyarray_new_from_data<'py, T>(
    py: Python<'py>,
    dims: &[usize],
    ty: NPY_TYPES,
    elem_size: Option<usize>,
    data: &[T],
) -> PyResult<Py<PyAny>> {
    let descr = unsafe { PY_ARRAY_API.PyArray_DescrFromType(py, ty as i32) };
    if descr.is_null() {
        return Err(PyErr::fetch(py));
    }
    if let Some(elem_size) = elem_size {
        unsafe {
            PyDataType_SET_ELSIZE(py, descr, elem_size as isize);
        }
    }
    let subtype = unsafe { PY_ARRAY_API.get_type_object(py, NpyTypes::PyArray_Type) };
    if subtype.is_null() {
        return Err(PyErr::fetch(py));
    }
    let pyarray = unsafe {
        PY_ARRAY_API.PyArray_NewFromDescr(
            py,
            subtype,
            descr,
            dims.len() as i32,
            dims.as_ptr() as *mut isize,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            npyffi::NPY_ARRAY_CARRAY,
            std::ptr::null_mut(),
        )
    };
    if pyarray.is_null() {
        return Err(PyErr::fetch(py));
    }
    unsafe {
        (*(pyarray as *mut PyArrayObject))
            .data
            .cast::<T>()
            .copy_from_nonoverlapping(data.as_ptr(), data.len());
    }
    let bound: Bound<'_, PyUntypedArray> =
        unsafe { Bound::from_owned_ptr(py, pyarray).downcast_into_unchecked() };
    bound.into_py_any(py)
}

unsafe fn eco_vec_from_numpy_array<T: Copy>(arr: *mut PyArrayObject, shape: &[usize]) -> EcoVec<T> {
    let len = shape.iter().copied().product::<usize>();
    unsafe { ecovec_from_slice(std::slice::from_raw_parts((*arr).data.cast(), len)) }
}

unsafe fn ecovec_from_slice<T: Copy>(data: &[T]) -> EcoVec<T> {
    #[repr(C)]
    struct TransmutedEcoVec<T> {
        ptr: std::ptr::NonNull<T>,
        len: usize,
        phantom: PhantomData<T>,
    }
    let mut vec = EcoVec::with_capacity(data.len());
    unsafe {
        let vec: &mut TransmutedEcoVec<T> = std::mem::transmute(&mut vec);
        std::ptr::copy_nonoverlapping(data.as_ptr(), vec.ptr.as_ptr(), data.len());
        vec.len = data.len();
    }
    vec
}
