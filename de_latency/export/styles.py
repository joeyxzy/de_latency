"""Display names and colors for GPU phases."""
def describe_gpu_phase(phase_name):
    styles = {
        "Preprocess": ("Worker Preprocess", "python", "good"),
        "Forward": ("Model Forward", "python", "olive"),
        "Postprocess": ("Postprocess", "python", "rail_load"),
        "Sample": ("Sampling", "python", "mauve"),
        "GPUModelRunner sample_tokens": ("GPUModelRunner sample_tokens", "python", "mauve"),
        "Bookkeep": ("Bookkeeping/Sync", "python", "cyan"),
        "Draft": ("Draft", "python", "yellow"),
        "EPLB": ("EPLB", "python", "rail_idle"),
        "ModelRunnerOutput": ("Model Runner Output", "python", "rail_response"),
    }
    return styles.get(
        phase_name,
        (f"GPU Phase: {phase_name}", "python", "background"),
    )
