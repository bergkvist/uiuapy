mod utils;

#[pyo3::pymodule(name = "uiua")]
mod numpy_uiua {
    use pyo3::exceptions::PyException;
    use pyo3::prelude::*;
    use pyo3::types::PyTuple;
    use pyo3::{IntoPyObjectExt, create_exception};
    use uiua::{Compiler, SafeSys, Uiua};

    use crate::utils::{numpy_array_to_uiua_value, uiua_value_to_numpy_array};

    create_exception!(numpy_uiua, UiuaCompileError, PyException);
    create_exception!(numpy_uiua, UiuaRuntimeError, PyException);

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
                .map_err(|e| UiuaCompileError::new_err(e.to_string()))?
                .finish();

            Ok(Self {
                assembly,
                spawn_threads,
            })
        }

        #[pyo3(signature = (*args))]
        pub fn __call__<'py>(&self, py: Python<'py>, args: Vec<Py<PyAny>>) -> PyResult<Py<PyAny>> {
            let mut uiua = Uiua::with_backend(match self.spawn_threads {
                true => SafeSys::with_thread_spawning(),
                false => SafeSys::new(),
            });
            let inputs = args
                .into_iter()
                .map(|x| numpy_array_to_uiua_value(py, &x))
                .collect::<PyResult<Vec<_>>>()?;
            uiua.push_all(inputs);
            uiua.run_asm(self.assembly.clone())
                .map_err(|e| UiuaRuntimeError::new_err(e.to_string()))?;
            let stack = uiua.take_stack();
            let outputs = stack
                .into_iter()
                .map(|x| uiua_value_to_numpy_array(py, x))
                .collect::<PyResult<Vec<_>>>()?;
            if outputs.len() == 1 {
                Ok(outputs.into_iter().next().unwrap())
            } else {
                Ok(PyTuple::new(py, outputs.into_iter())?.into_py_any(py)?)
            }
        }
    }
}
