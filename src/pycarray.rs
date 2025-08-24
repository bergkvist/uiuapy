use numpy::PY_ARRAY_API;
use numpy::npyffi::{
    self, NPY_TYPES, NpyTypes, PyArray_Descr, PyArrayObject, PyDataType_ELSIZE,
    PyDataType_SET_ELSIZE,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub struct PyContiguousArray;

impl PyContiguousArray {
    pub fn new<'py, T>(
        py: Python<'py>,
        ty: NPY_TYPES,
        mut data: Vec<T>,
        dims: &[usize],
        elem_size: Option<usize>,
    ) -> PyResult<Bound<'py, Self>> {
        ensure_dims_consistent_with_len(&data, dims, elem_size)?;
        unsafe {
            let subtype = PY_ARRAY_API.get_type_object(py, NpyTypes::PyArray_Type);
            if subtype.is_null() {
                return Err(PyErr::fetch(py));
            }

            let descr = PY_ARRAY_API.PyArray_DescrFromType(py, ty as i32);
            if descr.is_null() {
                return Err(PyErr::fetch(py));
            }
            if let Some(elem_size) = elem_size {
                PyDataType_SET_ELSIZE(py, descr, elem_size as isize);
                if PyErr::occurred(py) {
                    return Err(PyErr::fetch(py));
                }
            }

            let pyarray = PY_ARRAY_API.PyArray_NewFromDescr(
                py,
                subtype,
                descr,
                dims.len() as i32,
                dims.as_ptr() as *mut isize,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
            );
            if pyarray.is_null() {
                return Err(PyErr::fetch(py));
            }

            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                (*pyarray.cast::<PyArrayObject>()).data.cast(),
                data.len(),
            );
            // Prevents double drop of individual pyarray items that don't implement Copy
            data.set_len(0);

            Ok(Bound::from_owned_ptr(py, pyarray).downcast_into_unchecked())
        }
    }

    pub fn from_pyany<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        unsafe {
            let arr = PY_ARRAY_API.PyArray_FromAny(
                obj.py(),
                obj.as_ptr(),
                std::ptr::null_mut(),
                0,
                0,
                npyffi::NPY_ARRAY_CARRAY_RO,
                std::ptr::null_mut(),
            );
            if arr.is_null() || PyErr::occurred(obj.py()) {
                return Err(PyErr::fetch(obj.py()));
            }
            let res = Bound::from_owned_ptr(obj.py(), arr).downcast_into_unchecked();
            if res.nd() == 0 && res.dtype() == NPY_TYPES::NPY_OBJECT {
                // This avoids infinite recursion when passing in for example a dict.
                return Err(PyValueError::new_err(format!(
                    "Unsupported uiua input: {res}"
                )));
            }
            Ok(res)
        }
    }
}

fn ensure_dims_consistent_with_len<T>(
    data: &[T],
    dims: &[usize],
    elem_size: Option<usize>,
) -> PyResult<()> {
    let entries_per_elem = elem_size.map(|x| x / std::mem::size_of::<T>()).unwrap_or(1);
    let expected_len = dims.iter().product::<usize>() * entries_per_elem;
    let got_len = data.len();
    if expected_len != got_len {
        return Err(PyValueError::new_err(format!(
            "Expected {expected_len} elements, got {got_len} (dims: {dims:?}, elem_size: {elem_size:?}, entries_per_elem: {entries_per_elem})",
        )));
    }
    Ok(())
}

impl<'py> PyContiguousArrayMethods<'py> for Bound<'py, PyContiguousArray> {
    fn data<T>(&self) -> &[T] {
        let data = self.as_arrayobject().data.cast();
        let len = self.len() * self.elsize() / size_of::<T>();
        unsafe { std::slice::from_raw_parts(data, len) }
    }

    fn len(&self) -> usize {
        self.dims().iter().product()
    }

    fn dims(&self) -> Vec<usize> {
        let array = self.as_arrayobject();
        (0..array.nd)
            .map(|i| unsafe { array.dimensions.add(i as usize).read() as usize })
            .collect()
    }

    fn nd(&self) -> usize {
        self.as_arrayobject().nd as usize
    }

    fn elsize(&self) -> usize {
        let descr = self.as_arrayobject().descr;
        unsafe { PyDataType_ELSIZE(self.py(), descr) as usize }
    }

    fn dtype(&self) -> NPY_TYPES {
        let type_num = self.descr().type_num;
        unsafe { std::mem::transmute(type_num) }
    }

    fn descr(&self) -> &PyArray_Descr {
        let descr = self.as_arrayobject().descr;
        assert!(!descr.is_null());
        unsafe { &*descr }
    }

    fn as_arrayobject(&self) -> &PyArrayObject {
        let ptr = self.as_ptr().cast::<PyArrayObject>();
        assert!(!ptr.is_null());
        unsafe { &*ptr }
    }

    fn into_dtype(self, dtype: NPY_TYPES) -> PyResult<Bound<'py, PyContiguousArray>> {
        unsafe {
            let descr = PY_ARRAY_API.PyArray_DescrFromType(self.py(), dtype as i32);
            if descr.is_null() {
                return Err(PyErr::fetch(self.py()));
            }
            let array = PY_ARRAY_API.PyArray_FromArray(
                self.py(),
                self.as_ptr().cast::<PyArrayObject>(),
                descr,
                npyffi::NPY_ARRAY_CARRAY_RO,
            );
            if array.is_null() {
                return Err(PyErr::fetch(self.py()));
            }
            Ok(Bound::from_owned_ptr(self.py(), array).downcast_into_unchecked())
        }
    }

    fn return_value(self) -> PyResult<Bound<'py, PyAny>> {
        let py = self.py();
        let mp = self.into_ptr().cast();
        unsafe {
            let value = PY_ARRAY_API.PyArray_Return(py, mp);
            if value.is_null() {
                return Err(PyErr::fetch(py));
            }
            Ok(Bound::from_owned_ptr(py, value))
        }
    }
}

pub trait PyContiguousArrayMethods<'py> {
    fn data<T>(&self) -> &[T];
    fn len(&self) -> usize;
    fn dims(&self) -> Vec<usize>;
    fn nd(&self) -> usize;
    fn elsize(&self) -> usize;
    fn dtype(&self) -> NPY_TYPES;
    fn descr(&self) -> &PyArray_Descr;
    fn as_arrayobject(&self) -> &PyArrayObject;
    /// Attempts to convert a PyContiguousArray to a different dtype
    fn into_dtype(self, dtype: NPY_TYPES) -> PyResult<Bound<'py, PyContiguousArray>>;
    /// Converts 0-dimensional arrays into scalars by calling PyArray_Return
    fn return_value(self) -> PyResult<Bound<'py, PyAny>>;
}
