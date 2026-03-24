use pyo3::prelude::*;

mod encoding;
mod inference_pipeline;
mod pod5_io;
mod signal_refine;
mod signal_stats;

#[pymodule]
fn leech_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(signal_refine::seq_banded_dp, m)?)?;
    m.add_function(wrap_pyfunction!(signal_refine::extract_levels, m)?)?;
    m.add_function(wrap_pyfunction!(signal_refine::rough_rescale, m)?)?;
    m.add_function(wrap_pyfunction!(encoding::encode_signal_kmer, m)?)?;
    m.add_function(wrap_pyfunction!(signal_stats::compute_signal_stats, m)?)?;
    m.add_function(wrap_pyfunction!(pod5_io::read_pod5_batch, m)?)?;
    m.add_function(wrap_pyfunction!(pod5_io::preload_pod5_signals, m)?)?;
    m.add_class::<pod5_io::PreloadedSignals>()?;
    m.add_function(wrap_pyfunction!(inference_pipeline::extract_inference_chunks, m)?)?;
    m.add_function(wrap_pyfunction!(
        inference_pipeline::extract_chunks_from_preloaded,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(inference_pipeline::_test_process_read, m)?)?;
    m.add_function(wrap_pyfunction!(inference_pipeline::_test_ref_to_signal, m)?)?;
    m.add_function(wrap_pyfunction!(
        inference_pipeline::extract_training_chunks,
        m
    )?)?;
    Ok(())
}
