use numpy::npyffi::{NPY_TYPES, NpyTypes, PyArrayObject, PyDataType_ELSIZE, PyDataType_SET_ELSIZE};
use numpy::{PY_ARRAY_API, PyUntypedArray, npyffi};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyValueError;
use pyo3::ffi::Py_IS_TYPE;
use pyo3::prelude::*;
use uiua::{Boxed, Complex, Value};

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
        let dims = pyarray_dims(arr);
        let ty: NPY_TYPES = std::mem::transmute((*(*arr).descr).type_num);
        match ty {
            NPY_TYPES::NPY_UBYTE => {
                let slice = pyarray_data::<u8>(arr, &dims);
                Ok(Value::Byte(uiua::Array::<u8>::new(dims, slice)))
            }
            NPY_TYPES::NPY_DOUBLE => {
                let slice = pyarray_data::<f64>(arr, &dims);
                Ok(Value::Num(uiua::Array::<f64>::new(dims, slice)))
            }
            NPY_TYPES::NPY_CDOUBLE => {
                let slice = pyarray_data::<Complex>(arr, &dims);
                Ok(Value::Complex(uiua::Array::<Complex>::new(dims, slice)))
            }
            NPY_TYPES::NPY_UNICODE => {
                let mut dims = dims;
                dims.push(PyDataType_ELSIZE(py, (*arr).descr) as usize >> 2);
                let slice = pyarray_data::<char>(arr, &dims);
                Ok(Value::Char(uiua::Array::<char>::new(dims, slice)))
            }
            NPY_TYPES::NPY_OBJECT => {
                let slice = pyarray_data::<Py<PyAny>>(arr, &dims);
                let values = slice
                    .iter()
                    .map(|x| numpy_array_to_uiua_value(py, x).map(Boxed))
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(Value::Box(uiua::Array::<Boxed>::new(
                    dims,
                    values.as_slice(),
                )))
            }
            NPY_TYPES::NPY_BOOL => {
                let slice = pyarray_data::<bool>(arr, &dims);
                let slice = slice.iter().copied().map(|x| x as u8).collect::<Vec<_>>();
                Ok(Value::Byte(uiua::Array::<u8>::new(dims, slice.as_slice())))
            }
            NPY_TYPES::NPY_LONG => {
                let slice = pyarray_data::<i64>(arr, &dims);
                let slice = slice.iter().copied().map(|x| x as f64).collect::<Vec<_>>();
                Ok(Value::Num(uiua::Array::<f64>::new(dims, slice.as_slice())))
            }
            NPY_TYPES::NPY_ULONG => {
                let slice = pyarray_data::<u64>(arr, &dims);
                let slice = slice.iter().copied().map(|x| x as f64).collect::<Vec<_>>();
                Ok(Value::Num(uiua::Array::<f64>::new(dims, slice.as_slice())))
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
