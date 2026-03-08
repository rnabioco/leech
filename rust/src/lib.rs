use pyo3::prelude::*;

mod encoding;
mod signal_refine;

#[pymodule]
fn leech_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(signal_refine::seq_banded_dp, m)?)?;
    m.add_function(wrap_pyfunction!(signal_refine::extract_levels, m)?)?;
    m.add_function(wrap_pyfunction!(signal_refine::rough_rescale, m)?)?;
    m.add_function(wrap_pyfunction!(encoding::encode_signal_kmer, m)?)?;
    Ok(())
}
