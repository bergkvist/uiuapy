mod conversions;
mod ecovec;
mod pycarray;

#[pyo3::pymodule(name = "uiua")]
mod uiuapy {
    use pyo3::create_exception;
    use pyo3::exceptions::PyException;
    use pyo3::prelude::*;
    use pyo3::types::PyTuple;
    use uiua::{Compiler, SafeSys, Uiua};

    use crate::conversions::{numpy_to_uiua, uiua_to_numpy};

    create_exception!(Uiua, CompileError, PyException);
    create_exception!(Uiua, RuntimeError, PyException);

    #[pyclass(name = "compile")]
    pub struct Program {
        assembly: uiua::Assembly,
        allow_threads: Option<bool>,
    }

    #[pymethods]
    impl Program {
        #[new]
        #[pyo3(signature = (src, allow_threads=None))]
        pub fn new(src: &str, allow_threads: Option<bool>) -> PyResult<Self> {
            let mut compiler = Compiler::new();
            let assembly = compiler
                .load_str(src)
                .map_err(|e| CompileError::new_err(e.to_string()))?
                .finish();

            Ok(Self {
                assembly,
                allow_threads,
            })
        }

        #[pyo3(signature = (*args, allow_threads=None))]
        pub fn __call__<'py>(
            &self,
            py: Python<'py>,
            args: Vec<Bound<'py, PyAny>>,
            allow_threads: Option<bool>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let allow_threads = allow_threads.or(self.allow_threads).unwrap_or(false);
            let mut uiua = Uiua::with_backend(match allow_threads {
                true => SafeSys::with_thread_spawning(),
                false => SafeSys::new(),
            });
            let inputs = args
                .into_iter()
                .rev()
                .map(|x| numpy_to_uiua(&x))
                .collect::<PyResult<Vec<_>>>()?;
            uiua.push_all(inputs);
            uiua.run_asm(self.assembly.clone())
                .map_err(|e| RuntimeError::new_err(e.to_string()))?;
            let stack = uiua.take_stack();
            let outputs = stack
                .into_iter()
                .rev()
                .map(|x| uiua_to_numpy(py, &x))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(match outputs.len() {
                1 => outputs.into_iter().next().unwrap().into_any(),
                _ => PyTuple::new(py, outputs)?.into_any(),
            })
        }
    }
}
