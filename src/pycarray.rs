use numpy::npyffi::PyDataType_ELSIZE;
use numpy::npyffi::{NPY_TYPES, NpyTypes, PyArrayObject, PyDataType_SET_ELSIZE};
use numpy::{PY_ARRAY_API, npyffi};
use pyo3::exceptions::PyValueError;
use pyo3::ffi::Py_IS_TYPE;
use pyo3::prelude::*;

pub struct PyCArray;

impl PyCArray {
    pub fn new<'py, T>(
        py: Python<'py>,
        dims: &[usize],
        ty: NPY_TYPES,
        elem_size: Option<usize>,
        data: &[T],
    ) -> PyResult<Bound<'py, Self>> {
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
            (*pyarray.cast::<PyArrayObject>())
                .data
                .cast::<T>()
                .copy_from_nonoverlapping(data.as_ptr(), data.len());
        }
        Ok(unsafe { Bound::from_owned_ptr(py, pyarray).downcast_into_unchecked() })
    }

    pub fn try_from_ref<'py, 'a>(obj: &'a Bound<'py, PyAny>) -> PyResult<&'a Bound<'py, Self>> {
        unsafe {
            if Py_IS_TYPE(
                obj.as_ptr(),
                PY_ARRAY_API.get_type_object(obj.py(), NpyTypes::PyArray_Type),
            ) == 0
            {
                return Err(PyValueError::new_err(format!("{obj} is not a numpy array")));
            }

            let arr = obj.as_ptr().cast::<PyArrayObject>();
            if (*arr).flags & npyffi::NPY_ARRAY_ALIGNED != npyffi::NPY_ARRAY_ALIGNED {
                return Err(PyValueError::new_err(format!(
                    "{obj} must be properly aligned"
                )));
            }
            if (*arr).flags & npyffi::NPY_ARRAY_C_CONTIGUOUS != npyffi::NPY_ARRAY_C_CONTIGUOUS {
                return Err(PyValueError::new_err(format!("{obj} must be C-contiguous")));
            }
            Ok(obj.downcast_unchecked())
        }
    }
}

impl<'py> PyCArrayMethods for Bound<'py, PyCArray> {
    fn data<T>(&self) -> &[T] {
        let data = self.as_pyarray_ref().data.cast();
        let len = self.len();
        unsafe { std::slice::from_raw_parts(data, len) }
    }

    fn len(&self) -> usize {
        self.dims().iter().product()
    }

    fn dims(&self) -> Vec<usize> {
        let array = self.as_pyarray_ref();
        (0..array.nd)
            .map(|i| unsafe { array.dimensions.add(i as usize).read() } as usize)
            .collect()
    }

    fn elsize(&self) -> usize {
        let descr = self.as_pyarray_ref().descr;
        unsafe { PyDataType_ELSIZE(self.py(), descr) as usize }
    }

    fn dtype(&self) -> NPY_TYPES {
        let descr = self.as_pyarray_ref().descr;
        unsafe {
            let type_num = (*descr).type_num;
            std::mem::transmute(type_num)
        }
    }

    fn as_pyarray_ref(&self) -> &PyArrayObject {
        unsafe {
            self.as_ptr()
                .cast::<PyArrayObject>()
                .as_ref()
                .unwrap_unchecked()
        }
    }
}

pub trait PyCArrayMethods {
    fn data<T>(&self) -> &[T];
    fn len(&self) -> usize;
    fn dims(&self) -> Vec<usize>;
    fn elsize(&self) -> usize;
    fn dtype(&self) -> NPY_TYPES;
    fn as_pyarray_ref(&self) -> &PyArrayObject;
}
