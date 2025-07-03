#[pyo3::pymodule]
mod numpy_uiua {
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use pyo3::prelude::*;
    use uiua::{Compiler, SafeSys, Uiua};

    use crate::utils::uiua_value_to_numpy_array;

    #[pyclass]
    pub struct Program {
        assembly: uiua::Assembly,
    }

    #[pymethods]
    impl Program {
        #[new]
        pub fn new(src: &str) -> PyResult<Self> {
            let mut compiler = Compiler::new();
            let assembly = compiler
                .load_str(src)
                .map_err(|e| PyValueError::new_err(format!("Uiua compilation error: {e}")))?
                .finish();
            Ok(Self { assembly })
        }

        pub fn __call__<'py>(&self, py: Python<'py>) -> PyResult<Vec<Py<PyAny>>> {
            let mut uiua = Uiua::with_backend(SafeSys::with_thread_spawning());
            uiua.run_asm(self.assembly.clone())
                .map_err(|e| PyRuntimeError::new_err(format!("Uiua runtime error: {e}")))?;
            let stack = uiua.take_stack();
            stack
                .into_iter()
                .map(|x| uiua_value_to_numpy_array(py, x))
                .collect::<PyResult<Vec<_>>>()
        }
    }
}

pub mod utils {
    use numpy::npyffi::{NPY_TYPES, NpyTypes, PyArrayObject, PyDataType_SET_ELSIZE};
    use numpy::{
        Complex64, IxDyn, PY_ARRAY_API, PyArrayDyn, PyArrayMethods, PyUntypedArray, npyffi,
    };
    use pyo3::IntoPyObjectExt;
    use pyo3::prelude::*;
    use uiua::{Boxed, Value};

    pub fn uiua_value_to_numpy_array<'py>(py: Python<'py>, value: Value) -> PyResult<Py<PyAny>> {
        match value {
            Value::Num(values) => {
                let dims = IxDyn(values.shape.iter().cloned().collect::<Vec<_>>().as_slice());
                let data = values.elements().copied().collect::<Vec<_>>();
                unsafe {
                    let arr = PyArrayDyn::<f64>::new(py, dims, false);
                    arr.data()
                        .copy_from_nonoverlapping(data.as_ptr(), data.len());
                    arr.into_py_any(py)
                }
            }
            Value::Byte(values) => {
                let dims = IxDyn(values.shape.iter().cloned().collect::<Vec<_>>().as_slice());
                let data = values.elements().copied().collect::<Vec<_>>();
                unsafe {
                    let arr = PyArrayDyn::<u8>::new(py, dims, false);
                    arr.data()
                        .copy_from_nonoverlapping(data.as_ptr(), data.len());
                    arr.into_py_any(py)
                }
            }
            Value::Complex(values) => {
                let dims = IxDyn(values.shape.iter().cloned().collect::<Vec<_>>().as_slice());
                let data = values.elements().copied().collect::<Vec<_>>();
                unsafe {
                    let arr = PyArrayDyn::<Complex64>::new(py, dims, false);
                    arr.data()
                        .copy_from_nonoverlapping(data.as_ptr().cast::<Complex64>(), data.len());
                    arr.into_py_any(py)
                }
            }
            Value::Char(values) => {
                let nd = values.shape.iter().count() - 1;
                let data = values.elements().copied().collect::<Box<[_]>>();
                let mut dims = values.shape.into_iter().collect::<Box<[_]>>();
                unsafe {
                    let subtype = PY_ARRAY_API.get_type_object(py, NpyTypes::PyArray_Type);
                    let descr =
                        PY_ARRAY_API.PyArray_DescrFromType(py, NPY_TYPES::NPY_UNICODE as i32);
                    if descr.is_null() {
                        return Err(PyErr::fetch(py));
                    }
                    PyDataType_SET_ELSIZE(py, descr, 4 * *dims.last().unwrap_or(&0) as isize);
                    let arr = PY_ARRAY_API.PyArray_NewFromDescr(
                        py,
                        subtype,
                        descr,
                        nd as i32,
                        dims.as_mut_ptr() as *mut isize,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        npyffi::NPY_ARRAY_CARRAY,
                        std::ptr::null_mut(),
                    );
                    if arr.is_null() {
                        return Err(PyErr::fetch(py));
                    }
                    (*(arr as *mut PyArrayObject))
                        .data
                        .cast::<char>()
                        .copy_from_nonoverlapping(data.as_ptr(), data.len());

                    let bound: Bound<'_, PyUntypedArray> =
                        Bound::from_owned_ptr(py, arr).downcast_into_unchecked();
                    bound.into_py_any(py)
                }
            }
            Value::Box(values) => {
                let dims = IxDyn(values.shape.iter().cloned().collect::<Vec<_>>().as_slice());
                let data = values
                    .elements()
                    .map(|Boxed(value)| uiua_value_to_numpy_array(py, value.clone()))
                    .collect::<PyResult<Vec<_>>>()?;
                unsafe {
                    let arr = PyArrayDyn::<Py<PyAny>>::new(py, dims, false);
                    arr.data()
                        .copy_from_nonoverlapping(data.as_ptr(), data.len());
                    arr.into_py_any(py)
                }
            }
        }
    }
}
