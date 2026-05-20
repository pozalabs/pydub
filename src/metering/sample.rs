use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) trait Sample: Copy + Sized {
    fn to_f64(self) -> f64;
    fn max_amplitude() -> f64;
}

impl Sample for i8 {
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn max_amplitude() -> f64 {
        128.0
    }
}

impl Sample for i16 {
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn max_amplitude() -> f64 {
        32768.0
    }
}

impl Sample for i32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn max_amplitude() -> f64 {
        2147483648.0
    }
}

pub(crate) unsafe fn cast_samples<T: Sample>(data: &[u8]) -> &[T] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const T, data.len() / size_of::<T>()) }
}

pub(crate) fn validate_data(data: &[u8], sample_width: u8) -> PyResult<()> {
    if !matches!(sample_width, 1 | 2 | 4) {
        return Err(PyValueError::new_err(format!(
            "sample_width must be 1, 2, or 4 (got {sample_width})"
        )));
    }
    if data.len() % sample_width as usize != 0 {
        return Err(PyValueError::new_err(
            "data length is not a multiple of sample_width",
        ));
    }
    Ok(())
}
