use ecow::EcoVec;
use numpy::npyffi::NPY_TYPES;

use crate::pycarray::{PyCArray, PyCArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use uiua::{Boxed, Value};

mod ecovec;
mod pycarray;

#[pyo3::pymodule(name = "uiua")]
mod numpy_uiua {
    use pyo3::create_exception;
    use pyo3::exceptions::PyException;
    use pyo3::prelude::*;
    use pyo3::types::PyTuple;
    use uiua::{Compiler, SafeSys, Uiua};

    use super::{numpy_to_uiua, uiua_to_numpy};

    create_exception!(Uiua, CompileError, PyException);
    create_exception!(Uiua, RuntimeError, PyException);

    #[pyclass(name = "compile")]
    pub struct Program {
        assembly: uiua::Assembly,
        spawn_threads: bool,
    }

    #[pymethods]
    impl Program {
        #[new]
        #[pyo3(signature = (src, spawn_threads=false))]
        pub fn new(src: &str, spawn_threads: bool) -> PyResult<Self> {
            let mut compiler = Compiler::new();
            let assembly = compiler
                .load_str(src)
                .map_err(|e| CompileError::new_err(e.to_string()))?
                .finish();

            Ok(Self {
                assembly,
                spawn_threads,
            })
        }

        #[pyo3(signature = (*args))]
        pub fn __call__<'py>(
            &self,
            py: Python<'py>,
            args: Vec<Bound<'py, PyAny>>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let mut uiua = Uiua::with_backend(match self.spawn_threads {
                true => SafeSys::with_thread_spawning(),
                false => SafeSys::new(),
            });
            let inputs = args
                .into_iter()
                .map(|x| numpy_to_uiua(&x))
                .collect::<PyResult<Vec<_>>>()?;
            uiua.push_all(inputs);
            uiua.run_asm(self.assembly.clone())
                .map_err(|e| RuntimeError::new_err(e.to_string()))?;
            let stack = uiua.take_stack();
            let outputs = stack
                .into_iter()
                .map(|x| uiua_to_numpy(py, &x))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(match outputs.len() {
                1 => outputs.into_iter().next().unwrap().into_any(),
                _ => PyTuple::new(py, outputs)?.into_any(),
            })
        }
    }
}

pub fn numpy_to_uiua<'py>(array: &Bound<'py, PyAny>) -> PyResult<Value> {
    let arr = PyCArray::try_from_ref(array)?;
    let value = match arr.dtype() {
        NPY_TYPES::NPY_UBYTE => {
            Value::Byte(uiua::Array::new(arr.dims(), ecovec::from_slice(arr.data())))
        }
        NPY_TYPES::NPY_DOUBLE => {
            Value::Num(uiua::Array::new(arr.dims(), ecovec::from_slice(arr.data())))
        }
        NPY_TYPES::NPY_CDOUBLE => {
            Value::Complex(uiua::Array::new(arr.dims(), ecovec::from_slice(arr.data())))
        }
        NPY_TYPES::NPY_UNICODE => {
            let mut shape = arr.dims();
            shape.push(arr.elsize() / 4);
            Value::Char(uiua::Array::new(shape, ecovec::from_slice(arr.data())))
        }
        NPY_TYPES::NPY_OBJECT => {
            let data = arr
                .data::<Bound<'py, PyAny>>()
                .iter()
                .map(|x| numpy_to_uiua(x).map(Boxed))
                .collect::<PyResult<Vec<_>>>()?;
            Value::Box(uiua::Array::new(arr.dims(), EcoVec::from(data)))
        }
        NPY_TYPES::NPY_BOOL => Value::Byte(uiua::Array::new(
            arr.dims(),
            EcoVec::from(
                arr.data()
                    .iter()
                    .map(|x: &bool| *x as u8)
                    .collect::<Vec<_>>(),
            ),
        )),
        NPY_TYPES::NPY_FLOAT => Value::Num(uiua::Array::<f64>::new(
            arr.dims(),
            EcoVec::from(
                arr.data()
                    .iter()
                    .map(|x: &f32| *x as f64)
                    .collect::<Vec<_>>(),
            ),
        )),
        NPY_TYPES::NPY_ULONG => Value::Num(uiua::Array::<f64>::new(
            arr.dims(),
            EcoVec::from(
                arr.data()
                    .iter()
                    .map(|x: &u64| *x as f64)
                    .collect::<Vec<_>>(),
            ),
        )),
        NPY_TYPES::NPY_UINT => Value::Num(uiua::Array::<f64>::new(
            arr.dims(),
            EcoVec::from(
                arr.data()
                    .iter()
                    .map(|x: &u32| *x as f64)
                    .collect::<Vec<_>>(),
            ),
        )),
        NPY_TYPES::NPY_USHORT => Value::Num(uiua::Array::<f64>::new(
            arr.dims(),
            EcoVec::from(
                arr.data()
                    .iter()
                    .map(|x: &u16| *x as f64)
                    .collect::<Vec<_>>(),
            ),
        )),
        NPY_TYPES::NPY_LONG => Value::Num(uiua::Array::<f64>::new(
            arr.dims(),
            EcoVec::from(
                arr.data()
                    .iter()
                    .map(|x: &i64| *x as f64)
                    .collect::<Vec<_>>(),
            ),
        )),
        NPY_TYPES::NPY_INT => Value::Num(uiua::Array::<f64>::new(
            arr.dims(),
            EcoVec::from(
                arr.data()
                    .iter()
                    .map(|x: &i32| *x as f64)
                    .collect::<Vec<_>>(),
            ),
        )),
        NPY_TYPES::NPY_SHORT => Value::Num(uiua::Array::<f64>::new(
            arr.dims(),
            EcoVec::from(
                arr.data()
                    .iter()
                    .map(|x: &i16| *x as f64)
                    .collect::<Vec<_>>(),
            ),
        )),
        ty => {
            return Err(PyValueError::new_err(format!(
                "Unsupported numpy array type: {ty:?}"
            )));
        }
    };
    Ok(value)
}

pub fn uiua_to_numpy<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    let mut dims = value.shape.iter().copied().collect::<Vec<_>>();
    match value {
        Value::Num(values) => {
            let data = values.elements().copied().collect::<Vec<_>>();
            PyCArray::new(py, &dims, NPY_TYPES::NPY_DOUBLE, None, data)
        }
        Value::Byte(values) => {
            let data = values.elements().copied().collect::<Vec<_>>();
            PyCArray::new(py, &dims, NPY_TYPES::NPY_UBYTE, None, data)
        }
        Value::Complex(values) => {
            let data = values.elements().copied().collect::<Vec<_>>();
            PyCArray::new(py, &dims, NPY_TYPES::NPY_CDOUBLE, None, data)
        }
        Value::Char(values) => {
            let data = values.elements().copied().collect::<Vec<_>>();
            let elem_size = Some(4 * dims.pop().unwrap_or(0));
            PyCArray::new(py, &dims, NPY_TYPES::NPY_UNICODE, elem_size, data)
        }
        Value::Box(values) => {
            let data = values
                .elements()
                .map(|Boxed(value)| uiua_to_numpy(py, value))
                .collect::<PyResult<Vec<_>>>()?;
            PyCArray::new(py, &dims, NPY_TYPES::NPY_OBJECT, None, data)
        }
    }?
    .return_value()
}

pub fn timed<T>(label: &str, f: impl FnOnce() -> T) -> T {
    let t0 = std::time::Instant::now();
    let result = f();
    println!("{label}: {:?}", t0.elapsed());
    result
}
