use ecow::EcoVec;
use numpy::npyffi::NPY_TYPES;

use crate::ecovec;
use crate::pycarray::{PyContiguousArray, PyContiguousArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use uiua::{Boxed, Value};

pub fn uiua_to_numpy<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    let dims = value.shape.iter().copied().collect::<Vec<_>>();
    match value {
        Value::Num(values) => {
            let data = values.elements().copied().collect::<Vec<_>>();
            PyContiguousArray::new(py, NPY_TYPES::NPY_DOUBLE, data, &dims, None)
        }
        Value::Byte(values) => {
            let data = values.elements().copied().collect::<Vec<_>>();
            PyContiguousArray::new(py, NPY_TYPES::NPY_UBYTE, data, &dims, None)
        }
        Value::Complex(values) => {
            let data = values.elements().copied().collect::<Vec<_>>();
            PyContiguousArray::new(py, NPY_TYPES::NPY_CDOUBLE, data, &dims, None)
        }
        Value::Char(values) => {
            let data = values.elements().copied().collect::<Vec<_>>();
            let mut dims = dims;
            let elem_size = Some(size_of::<char>() * dims.pop().unwrap_or(0));
            PyContiguousArray::new(py, NPY_TYPES::NPY_UNICODE, data, &dims, elem_size)
        }
        Value::Box(values) => {
            let data = values
                .elements()
                .map(|Boxed(value)| uiua_to_numpy(py, value))
                .collect::<PyResult<Vec<_>>>()?;
            PyContiguousArray::new(py, NPY_TYPES::NPY_OBJECT, data, &dims, None)
        }
    }?
    .return_value()
}

pub fn numpy_to_uiua<'py>(array: &Bound<'py, PyAny>) -> PyResult<Value> {
    let arr = PyContiguousArray::from_pyany(array)?;
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
            shape.push(arr.elsize() / size_of::<char>());
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
