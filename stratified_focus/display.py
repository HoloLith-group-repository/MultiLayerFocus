"""Shared wording and styling for unavailable optional visualization panels."""
WARNING_STYLE = 'color: #526477; font-size: 14px;'


def calculation_warning(name, option):
    return f'{name} was not calculated.\nSelect “{option}” below to calculate it, or enable it before running a simulation.'


def preview_matches_result(preview, result):
    """Compare confirmed inputs and pupil values; output-selection flags are not inputs."""
    if preview is None:
        return True
    if result is None:
        return False
    from dataclasses import replace
    import numpy as np
    def inputs(config):
        return replace(config,sampling=replace(config.sampling,compute_axial=False,compute_volume=False))
    if inputs(preview.config) != inputs(result.config):
        return False
    return all(np.array_equal(getattr(preview,name),getattr(result,name))
               for name in ('pupil_ex','pupil_ey','pupil_x','pupil_y'))
