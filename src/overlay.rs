use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::utils::define_gain;

macro_rules! define_mix {
    ($fn_name:ident, $sample_type:ty, $wider_type:ty) => {
        fn $fn_name(a: $sample_type, b: $sample_type) -> $sample_type {
            let val = a as $wider_type + b as $wider_type;
            val.clamp(
                <$sample_type>::MIN as $wider_type,
                <$sample_type>::MAX as $wider_type,
            ) as $sample_type
        }
    };
}

define_gain!(gain_8, i8, i16);
define_gain!(gain_16, i16, i32);
define_gain!(gain_32, i32, i64);
define_mix!(mix_8, i8, i16);
define_mix!(mix_16, i16, i32);
define_mix!(mix_32, i32, i64);

#[pyfunction]
#[pyo3(signature = (seg1_data, seg2_data, sample_width, position, times, gain_during_overlay=0))]
pub fn overlay_segments<'py>(
    py: Python<'py>,
    seg1_data: &[u8],
    seg2_data: &[u8],
    sample_width: i32,
    position: i32,
    times: i32,
    gain_during_overlay: i32,
) -> PyResult<Bound<'py, PyBytes>> {
    let seg1_len = seg1_data.len() as i32;
    let seg2_len = seg2_data.len() as i32;

    if position >= seg1_len {
        return Ok(PyBytes::new(py, seg1_data));
    }

    if !matches!(sample_width, 1 | 2 | 4) {
        return Err(PyValueError::new_err(format!(
            "sample_width must be 1, 2, or 4 (got {sample_width})"
        )));
    }

    if seg1_len % sample_width != 0 {
        return Err(PyValueError::new_err(
            "seg1_data length is not a multiple of sample_width",
        ));
    }

    if seg2_len % sample_width != 0 {
        return Err(PyValueError::new_err(
            "seg2_data length is not a multiple of sample_width",
        ));
    }

    if sample_width > 1 && position % sample_width != 0 {
        return Err(PyValueError::new_err(format!(
            "position ({position}) must be aligned to sample_width ({sample_width})"
        )));
    }

    let apply_gain = gain_during_overlay != 0;
    let db_factor = if apply_gain {
        10.0_f64.powf(gain_during_overlay as f64 / 20.0)
    } else {
        1.0
    };

    let seg1_len_u = seg1_len as usize;
    let output = PyBytes::new_with(py, seg1_len_u, |out_buf| {
        out_buf.copy_from_slice(seg1_data);

        let repeat_to_fill = times < 0;
        let mut remaining_times = times;
        let seg1_len_after_pos = seg1_len - position;
        let mut current_position: i32 = 0;
        let position = position as usize;
        let seg2_len = seg2_len as usize;

        macro_rules! overlay_loop {
            ($sample_type:ty, $gain_fn:ident, $mix_fn:ident) => {
                while (repeat_to_fill || remaining_times > 0)
                    && current_position < seg1_len_after_pos
                {
                    let remaining = (seg1_len_after_pos - current_position) as usize;
                    let chunk_len = remaining.min(seg2_len);
                    let num_samples = chunk_len / std::mem::size_of::<$sample_type>();
                    let offset = position + current_position as usize;

                    let out_slice = unsafe {
                        std::slice::from_raw_parts_mut(
                            out_buf.as_mut_ptr().add(offset) as *mut $sample_type,
                            num_samples,
                        )
                    };
                    let s2_slice = unsafe {
                        std::slice::from_raw_parts(
                            seg2_data.as_ptr() as *const $sample_type,
                            num_samples,
                        )
                    };

                    for i in 0..num_samples {
                        if apply_gain {
                            out_slice[i] = $mix_fn($gain_fn(out_slice[i], db_factor), s2_slice[i]);
                        } else {
                            out_slice[i] = $mix_fn(out_slice[i], s2_slice[i]);
                        }
                    }

                    current_position += chunk_len as i32;
                    if !repeat_to_fill {
                        remaining_times -= 1;
                    }
                }
            };
        }

        match sample_width {
            1 => overlay_loop!(i8, gain_8, mix_8),
            2 => overlay_loop!(i16, gain_16, mix_16),
            4 => overlay_loop!(i32, gain_32, mix_32),
            _ => unreachable!(),
        }

        Ok(())
    })?;

    Ok(output)
}

#[pyfunction]
pub fn mix_segments<'py>(
    py: Python<'py>,
    segments: Vec<Bound<'py, PyBytes>>,
    sample_width: i32,
    positions: Vec<i32>,
) -> PyResult<Bound<'py, PyBytes>> {
    if segments.is_empty() {
        return Err(PyValueError::new_err("segments must not be empty"));
    }

    if segments.len() != positions.len() {
        return Err(PyValueError::new_err(format!(
            "'positions' length ({}) must match 'segments' length ({})",
            positions.len(),
            segments.len(),
        )));
    }

    if !matches!(sample_width, 1 | 2 | 4) {
        return Err(PyValueError::new_err(format!(
            "sample_width must be 1, 2, or 4 (got {sample_width})"
        )));
    }

    let sw = sample_width as usize;
    let slices: Vec<&[u8]> = segments.iter().map(|s| s.as_bytes()).collect();

    for (i, (s, &pos)) in slices.iter().zip(positions.iter()).enumerate() {
        if s.len() % sw != 0 {
            return Err(PyValueError::new_err(format!(
                "segment {i} length is not a multiple of sample_width"
            )));
        }
        if pos < 0 {
            return Err(PyValueError::new_err(format!(
                "position {i} must be non-negative (got {pos})"
            )));
        }
        if (pos as usize) % sw != 0 {
            return Err(PyValueError::new_err(format!(
                "position {i} ({pos}) is not a multiple of sample_width"
            )));
        }
    }

    let total_len = slices
        .iter()
        .zip(positions.iter())
        .map(|(s, &pos)| pos as usize + s.len())
        .max()
        .unwrap();

    let output = PyBytes::new_with(py, total_len, |out_buf| {
        mix_segments_raw(out_buf, &slices, &positions, sw);
        Ok(())
    })?;

    Ok(output)
}

fn mix_segments_raw(out_buf: &mut [u8], slices: &[&[u8]], positions: &[i32], sample_width: usize) {
    out_buf.fill(0);

    macro_rules! mix_at {
        ($sample_type:ty, $mix_fn:ident) => {
            for (seg, &pos) in slices.iter().zip(positions.iter()) {
                let offset_samples = pos as usize / std::mem::size_of::<$sample_type>();
                let num_samples = seg.len() / std::mem::size_of::<$sample_type>();
                let out_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        out_buf.as_mut_ptr() as *mut $sample_type,
                        out_buf.len() / std::mem::size_of::<$sample_type>(),
                    )
                };
                let seg_slice = unsafe {
                    std::slice::from_raw_parts(seg.as_ptr() as *const $sample_type, num_samples)
                };
                for j in 0..num_samples {
                    out_slice[offset_samples + j] =
                        $mix_fn(out_slice[offset_samples + j], seg_slice[j]);
                }
            }
        };
    }

    match sample_width {
        1 => mix_at!(i8, mix_8),
        2 => mix_at!(i16, mix_16),
        4 => mix_at!(i32, mix_32),
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gain_8_normal() {
        assert_eq!(gain_8(50, 2.0), 100);
        assert_eq!(gain_8(-50, 2.0), -100);
    }

    #[test]
    fn test_gain_8_clamp_overflow() {
        assert_eq!(gain_8(100, 2.0), 127);
    }

    #[test]
    fn test_gain_8_clamp_underflow() {
        assert_eq!(gain_8(-100, 2.0), -128);
    }

    #[test]
    fn test_mix_8_normal() {
        assert_eq!(mix_8(50, 30), 80);
        assert_eq!(mix_8(-50, -30), -80);
    }

    #[test]
    fn test_mix_8_clamp_overflow() {
        assert_eq!(mix_8(100, 100), 127);
    }

    #[test]
    fn test_mix_8_clamp_underflow() {
        assert_eq!(mix_8(-100, -100), -128);
    }

    #[test]
    fn test_gain_16_normal() {
        assert_eq!(gain_16(1000, 2.0), 2000);
        assert_eq!(gain_16(-1000, 2.0), -2000);
    }

    #[test]
    fn test_gain_16_clamp_overflow() {
        assert_eq!(gain_16(30000, 2.0), i16::MAX);
    }

    #[test]
    fn test_gain_16_clamp_underflow() {
        assert_eq!(gain_16(-30000, 2.0), i16::MIN);
    }

    #[test]
    fn test_mix_16_normal() {
        assert_eq!(mix_16(1000, 2000), 3000);
        assert_eq!(mix_16(-1000, -2000), -3000);
    }

    #[test]
    fn test_mix_16_clamp_overflow() {
        assert_eq!(mix_16(i16::MAX, 1), i16::MAX);
    }

    #[test]
    fn test_mix_16_clamp_underflow() {
        assert_eq!(mix_16(i16::MIN, -1), i16::MIN);
    }

    #[test]
    fn test_gain_32_normal() {
        assert_eq!(gain_32(100000, 2.0), 200000);
        assert_eq!(gain_32(-100000, 2.0), -200000);
    }

    #[test]
    fn test_gain_32_clamp_overflow() {
        assert_eq!(gain_32(2000000000, 2.0), i32::MAX);
    }

    #[test]
    fn test_gain_32_clamp_underflow() {
        assert_eq!(gain_32(-2000000000, 2.0), i32::MIN);
    }

    #[test]
    fn test_mix_32_normal() {
        assert_eq!(mix_32(100000, 200000), 300000);
        assert_eq!(mix_32(-100000, -200000), -300000);
    }

    #[test]
    fn test_mix_32_clamp_overflow() {
        assert_eq!(mix_32(i32::MAX, 1), i32::MAX);
    }

    #[test]
    fn test_mix_32_clamp_underflow() {
        assert_eq!(mix_32(i32::MIN, -1), i32::MIN);
    }
}
