use numpy::npyffi::PyDataType_ELSIZE;
use numpy::npyffi::{NPY_TYPES, NpyTypes, PyArrayObject, PyDataType_SET_ELSIZE};
use numpy::{PY_ARRAY_API, npyffi};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub struct PyCArray;

impl PyCArray {
    pub fn new<'py, T>(
        py: Python<'py>,
        dims: &[usize],
        ty: NPY_TYPES,
        elem_size: Option<usize>,
        mut data: Vec<T>,
    ) -> PyResult<Bound<'py, Self>> {
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
                npyffi::NPY_ARRAY_CARRAY,
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

    pub fn try_from_ref<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
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

impl<'py> PyCArrayMethods<'py> for Bound<'py, PyCArray> {
    fn data<T>(&self) -> &[T] {
        let data = self.as_pyarrayobject().data.cast();
        let len = self.len() * self.elsize() / size_of::<T>();
        unsafe { std::slice::from_raw_parts(data, len) }
    }

    fn len(&self) -> usize {
        self.dims().iter().product()
    }

    fn dims(&self) -> Vec<usize> {
        let array = self.as_pyarrayobject();
        (0..array.nd)
            .map(|i| unsafe { array.dimensions.add(i as usize).read() as usize })
            .collect()
    }

    fn nd(&self) -> usize {
        self.as_pyarrayobject().nd as usize
    }

    fn elsize(&self) -> usize {
        let descr = self.as_pyarrayobject().descr;
        unsafe { PyDataType_ELSIZE(self.py(), descr) as usize }
    }

    fn dtype(&self) -> NPY_TYPES {
        let descr = self.as_pyarrayobject().descr;
        assert!(!descr.is_null());
        unsafe {
            let type_num = (*descr).type_num;
            std::mem::transmute(type_num)
        }
    }

    fn as_pyarrayobject(&self) -> &PyArrayObject {
        unsafe {
            self.as_ptr()
                .cast::<PyArrayObject>()
                .as_ref()
                .unwrap_unchecked()
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

pub trait PyCArrayMethods<'py> {
    fn data<T>(&self) -> &[T];
    fn len(&self) -> usize;
    fn dims(&self) -> Vec<usize>;
    fn nd(&self) -> usize;
    fn elsize(&self) -> usize;
    fn dtype(&self) -> NPY_TYPES;
    fn as_pyarrayobject(&self) -> &PyArrayObject;
    /// Converts 0-dimensional arrays into scalars by calling PyArray_Return
    fn return_value(self) -> PyResult<Bound<'py, PyAny>>;
}
