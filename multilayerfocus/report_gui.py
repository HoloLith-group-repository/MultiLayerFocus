"""Detailed MultiLayerFocus report built from the current computed result and PyQtGraph tabs.

Copy this module into the ``multilayerfocus`` package. It does not rerun the solver.
"""
from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
import textwrap

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.image import imread
from PyQt6.QtCore import QBuffer, QIODevice
from PyQt6.QtWidgets import QApplication


def _fmt(value, unit=""):
    if value is None:
        return "not available"
    try:
        val = float(value)
        if not np.isfinite(val):
            return "not available"
        return f"{val:.6g}{unit}"
    except (TypeError, ValueError):
        return str(value)


def _summary(r):
    s, m, b = r.scene, (r.metrics or {}), (r.budget or {})
    result = []

    def section(title, entries):
        result.append((title.upper(), True))
        result.extend((str(entry), False) for entry in entries)
        result.append(("", False))

    section("Run information", [
        f"Generated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"Result type: {'Live preview' if r.preview else 'Full computation'}",
        f"Compute time: {_fmt(r.seconds, ' [s]')}",
        f"Integration method: {r.method}",
        f"Angular sampling: Ntheta = {r.ntheta}, Nphi (effective) = {r.nphi}",
        f"Estimated sampling error: {_fmt(r.sampling_error)}",
    ])

    section("Objective and illumination", [
        f"Objective: {s.objective}",
        f"Numerical aperture: {_fmt(s.na)}; focal length: {_fmt(s.focal_length, ' [µm]')}",
        f"Vacuum wavelength: {_fmt(s.wavelength, ' [µm]')}",
        f"Polarization: {s.polarization}; gamma = {_fmt(s.gamma)}, beta = {_fmt(s.beta)}, delta = {_fmt(s.delta)}",
        f"Vortex charge: {s.vortex}; pupil fill ratio: {_fmt(s.fill)} (0 = uniform)",
        f"Input power: {_fmt(s.power_mW, ' [mW]')}",
        "Imported pupil files: " + (
            ", ".join(Path(p).name for p in s.pupil_files)
            if s.pupil_files else "none; generated pupil"
        ),
        "Zernike phase coefficients [rad]: " + ", ".join(
            f"Z{i+1}={_fmt(v)}" for i, v in enumerate(s.zernike)
        ),
    ])

    layer_lines = []
    for i, layer in enumerate(s.layers, 1):
        thickness = "semi-infinite" if layer.d is None else _fmt(layer.d, " [µm]")
        layer_lines.append(
            f"Layer {i}: {layer.name}; n = {_fmt(layer.n_re)}; k = {_fmt(layer.n_im)}; thickness = {thickness}"
        )
    section("Optical stack (incident to exit)", layer_lines + [
        f"Last interface z_out: {_fmt(s.z_out, ' [µm]')}"
    ])

    section("Observation and accuracy", [
        f"XY plane z0: {_fmt(s.z0, ' [µm]')}; x0: {_fmt(s.x0, ' [µm]')}; y0: {_fmt(s.y0, ' [µm]')}",
        f"Lateral size: {_fmt(s.lateral, ' [µm]')}; axial size: {_fmt(s.axial, ' [µm]')}",
        f"Sampling: XY = {s.nxy} x {s.nxy}; axial Z = {s.nz}",
        f"Accuracy setting: {s.quality}; custom Ntheta = {s.ntheta}, Nphi = {s.nphi}",
        f"XYZ volume requested: {bool(s.volume)}; volume samples: {s.volume_xy} x {s.volume_xy} x {s.volume_z}",
        "Volume views are intentionally omitted from this PDF.",
    ])

    frac = m.get("frac")
    section("Computed field metrics", [
        f"Peak field amplitude |E|: {_fmt(m.get('E_peak_Vm'), ' [V/m]')}",
        f"Peak intensity: {_fmt(m.get('I_peak_Wcm2'), ' [W/cm²]')}",
        f"Peak axial position: {_fmt(m.get('z_peak'), ' [µm]')}",
        f"FWHM X: {_fmt(m.get('fwhm_x') * 1000 if m.get('fwhm_x') is not None else None, ' [nm]')}",
        f"FWHM Y: {_fmt(m.get('fwhm_y') * 1000 if m.get('fwhm_y') is not None else None, ' [nm]')}",
        f"FWHM Z: {_fmt(m.get('fwhm_z') * 1000 if m.get('fwhm_z') is not None else None, ' [nm]')}",
        "Integrated component fractions (Ex, Ey, Ez): " + (
            ", ".join(f"{100*float(v):.3g}%" for v in frac)
            if frac is not None else "not available"
        ),
    ])

    absorbs_raw = b.get("A")
    absorbs = list(absorbs_raw) if absorbs_raw is not None else []
    section("Beam energy budget", [
        f"Reflectance R: {_fmt(100 * b['R'] if 'R' in b else None, ' %')}",
        f"Transmittance T: {_fmt(100 * b['T'] if 'T' in b else None, ' %')}",
        f"Total absorbed fraction: {_fmt(100 * sum(absorbs), ' %')}",
    ] + [
        f"Absorption, layer {i+1} ({s.layers[i].name}): {_fmt(100*value, ' %')}"
        for i, value in enumerate(absorbs) if i < len(s.layers)
    ])
    section("Figure conventions", [
        "Figures reproduce the GUI plots at the currently selected quantities, positions, and display settings.",
        "Color scales may be normalized; read their individual labels before interpreting amplitude.",
        "Rectangular axial display areas may not preserve an equal physical scale in X/Y and Z.",
        "Graphs document a computed numerical model; accuracy depends on sampling and input material data.",
        "Parameter sweep, if shown, is the most recently run sweep and may use different inputs.",
    ])
    return result


def _save_summary(pdf, entries):
    """Create as many readable summary pages as required."""
    fig = None
    y = 0.0
    page = 0

    def new_page():
        nonlocal fig, y, page
        if fig is not None:
            pdf.savefig(fig)
        page += 1
        fig = Figure(figsize=(11.69, 8.27), facecolor="white")
        fig.text(.055, .945, "MultiLayerFocus  |  Detailed simulation report",
                 fontsize=17, fontweight="bold", color="#153d69")
        fig.text(.055, .91, f"Numerical inputs and outcomes  ·  Page {page}",
                 fontsize=9, color="#45586d")
        fig.text(.055, .035, "MultiLayerFocus  |  2D figures reproduced from the GUI",
                 fontsize=8, color="#6b7280")
        y = .875

    new_page()
    for content, heading in entries:
        rows = textwrap.wrap(content, width=128, break_long_words=False) if content else [""]
        for j, row in enumerate(rows):
            line_height = .028 if heading else .024
            if y < .09:
                new_page()
            fig.text(.06, y, row,
                     fontsize=10 if heading else 8.7,
                     fontweight="bold" if heading else "normal",
                     color="#153d69" if heading else "#1f2937")
            y -= line_height
        if heading:
            y -= .005
    if fig is not None:
        pdf.savefig(fig)


def _widget_image(widget):
    """Capture a Qt widget as a NumPy image, without changing its plot data."""
    pix = widget.grab()
    if pix.isNull():
        raise RuntimeError("The widget could not be captured")
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    if not pix.save(buffer, "PNG"):
        buffer.close()
        raise RuntimeError("Failed to encode the widget image")
    encoded = bytes(buffer.data())
    buffer.close()
    return imread(BytesIO(encoded), format="png")


def _save_figure_page(pdf, title, image, caption):
    fig = Figure(figsize=(11.69, 8.27), facecolor="white")
    fig.text(.055, .95, title, fontsize=16, fontweight="bold", color="#153d69")
    fig.text(.055, .905, caption, fontsize=9, color="#45586d")
    ax = fig.add_axes([.035, .065, .93, .81])
    ax.imshow(image, interpolation="nearest")
    ax.axis("off")
    ax.set_aspect("equal", adjustable="box")
    pdf.savefig(fig)


def export_detailed_report(window, path):
    """Export run parameters, metrics, non-volume GUI plots, and A/B/N maps."""
    r = window.run_result
    if r is None:
        raise ValueError("Compute a simulation before exporting the report")

    tabs = window.tabs
    old_tab = tabs.currentIndex()
    old_abn = getattr(window, "pol_abn_dialog", None)
    focus_selector = getattr(window, "focus_axial_plane", None)
    old_focus_plane = focus_selector.currentIndex() if focus_selector is not None else None
    was_abn_visible = bool(old_abn is not None and old_abn.isVisible())

    with PdfPages(path) as pdf:
        _save_summary(pdf, _summary(r))
        try:
            for index in range(tabs.count()):
                title = tabs.tabText(index)
                low_title = title.lower()

                # No XYZ volumes or VTK figures in the report.
                if "volume" in low_title or low_title == "3d":
                    continue
                if "sweep" in low_title and not getattr(window, "sw_x", []):
                    continue

                tabs.setCurrentIndex(index)

                # Include both axial orientations of Focal field (XZ and YZ).
                orientations = [None]
                if low_title == "focal field" and focus_selector is not None:
                    orientations = list(range(focus_selector.count()))

                for orientation in orientations:
                    page_title = title
                    if orientation is not None:
                        focus_selector.setCurrentIndex(orientation)
                        page_title += f"  |  {focus_selector.currentText()}"

                    window.render()  # Cached results; no new Compute
                    QApplication.processEvents()
                    image = _widget_image(tabs.widget(index))
                    _save_figure_page(
                        pdf, page_title, image,
                        "Current 2D GUI figures. Read each plot's axis units and color scale."
                    )

            # The A/B/N 3x3 display is a separate window, not a tab.
            if hasattr(window, "_show_abn_dialog"):
                window._show_abn_dialog()
                QApplication.processEvents()
                dialog = getattr(window, "pol_abn_dialog", None)
                if dialog is not None:
                    _save_figure_page(
                        pdf, "Polarization ellipse components: A, B, N",
                        _widget_image(dialog),
                        "Major axis A, minor axis B, and normal N; Cartesian components."
                    )
        finally:
            if focus_selector is not None and old_focus_plane is not None:
                focus_selector.setCurrentIndex(old_focus_plane)
            tabs.setCurrentIndex(old_tab)
            window.render()
            if not was_abn_visible:
                dialog = getattr(window, "pol_abn_dialog", None)
                if dialog is not None:
                    dialog.hide()
            QApplication.processEvents()
