"""MultiLayerFocus main window."""
from __future__ import annotations
import sys, os, json, copy, math, time, traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QSettings, QRectF
from PyQt6.QtGui import QIcon, QAction, QColor, QPixmap, QPainter, QFont, QGuiApplication, QPen
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
                             QPushButton, QComboBox, QCheckBox, QScrollArea, QTabWidget, QSplitter, QTableWidget,
                             QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox, QProgressBar, QDialog,
                             QDialogButtonBox, QFormLayout, QMenu, QToolButton, QSizePolicy, QStatusBar, QFrame,
                             QAbstractItemView, QAbstractSpinBox, QInputDialog)
from .engine import pupil_for
from stratified_focus.pupil import generate_inc_field, critical_angles
from . import __version__, presets
from .engine import Scene, Layer, run, QUANTITIES, quantity, fwhm, auto_sampling
from .theme import stylesheet, PALETTES
from .widgets import (Card, Spin, ISpin, SliderSpin, row, MetricTile, MapView, LinePlot, StackDiagram, CMAPS,
                      get_cmap)
from stratified_focus.multilayer import power_coefficients
from stratified_focus.analysis import polarization_ellipse_3d
from scipy.interpolate import RegularGridInterpolator
from .plot_interaction import UnifiedViewBox, setup_unified_plot
from PyQt6.QtWidgets import QSlider

ICON = Path(__file__).with_name('icon.png')


def _theme_toggle_icon(theme):
    """Draw a readable 25-pixel half-filled circle independent of system fonts."""
    pixmap = QPixmap(48, 48)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    ink = QColor(PALETTES[theme]['text'])
    circle = QRectF(8, 8, 32, 32)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(ink, 3))
    painter.drawEllipse(circle)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(ink)
    painter.drawPie(circle, 90 * 16, 180 * 16)
    painter.end()
    return QIcon(pixmap)


# ======================================================================== workers
class Worker(QThread):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(float)

    def __init__(self, scene, preview, check=False):
        super().__init__()
        self.scene = copy.deepcopy(scene)
        self.preview = preview
        self.check = check
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            r = run(self.scene, preview=self.preview, check=self.check,
                    progress=lambda f: self.progress.emit(f), cancel=lambda: self._stop)
            if not self._stop:
                self.done.emit(r)
        except Exception as exc:
            if not self._stop:
                self.failed.emit(f'{type(exc).__name__}: {exc}')


class SweepWorker(QThread):
    point = pyqtSignal(float, object)
    failed = pyqtSignal(str)
    finished_all = pyqtSignal()

    def __init__(self, scene, setter, values, preview):
        super().__init__()
        self.scene, self.setter, self.values, self.preview = copy.deepcopy(scene), setter, values, preview
        self._stop = False

    def stop(self): self._stop = True

    def run(self):
        try:
            for v in self.values:
                if self._stop: break
                s = copy.deepcopy(self.scene); self.setter(s, v)
                r = run(s, preview=self.preview, want_h=False)
                self.point.emit(float(v), dict(metrics=r.metrics, budget=r.budget))
            self.finished_all.emit()
        except Exception as exc:
            self.failed.emit(f'{type(exc).__name__}: {exc}')


# ======================================================================== dialogs
class ZernikeDialog(QDialog):
    def __init__(self, values, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Pupil aberrations (Zernike, radians)')
        lay = QVBoxLayout(self)
        note = QLabel('Unnormalized real Zernike polynomials Z(n,m) of the normalized pupil radius; '
                      'coefficients are phases in radians.')
        note.setWordWrap(True); note.setObjectName('muted'); lay.addWidget(note)
        form = QFormLayout(); self.spins = []
        for name, v in zip(presets.ZERNIKE_NAMES, values):
            s = Spin(v, -50, 50, 3, 0.1, ' [rad]'); s.setFixedWidth(110); form.addRow(name, s); self.spins.append(s)
        lay.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel |
                              QDialogButtonBox.StandardButton.Reset)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        bb.button(QDialogButtonBox.StandardButton.Reset).clicked.connect(lambda: [s.setValue(0) for s in self.spins])
        lay.addWidget(bb)

    def values(self):
        return [s.value() for s in self.spins]





# ======================================================================== stack editor
class StackEditor(QWidget):
    changed = pyqtSignal()
    COLS = ['Material', 'n', 'k', 'd [µm]']

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(6)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(self.COLS)
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in (1, 2, 3):
            h.setSectionResizeMode(c, QHeaderView.ResizeMode.Fixed); self.table.setColumnWidth(c, 58)
        self.table.verticalHeader().setFixedWidth(58)
        self.table.setCornerButtonEnabled(False)
        self.table.verticalHeader().setDefaultSectionSize(28)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setMinimumHeight(176)
        lay.addWidget(self.table)
        bar = QHBoxLayout(); bar.setSpacing(4)
        for text, slot, tip in (('＋ Layer', self.add_layer, 'Insert a finite layer above the selected medium'),
                                ('－', self.remove_layer, 'Remove the selected medium'),
                                ('↑', lambda: self.move(+1), 'Move the selected layer towards the exit side'),
                                ('↓', lambda: self.move(-1), 'Move the selected layer towards the objective')):
            b = QPushButton(text); b.setToolTip(tip); b.clicked.connect(slot); bar.addWidget(b)
        bar.addStretch(1)
        lay.addLayout(bar)
        self.table.itemChanged.connect(self._item_changed)
        self._block = False
        self.layers = []

    # table rows are shown exit medium on top, incident at the bottom (as in the microscope)
    def set_layers(self, layers):
        self._block = True
        self.layers = [copy.deepcopy(l) for l in layers]
        for r in range(self.table.rowCount()):
            w = self.table.cellWidget(r, 0)
            if w is not None:
                w.blockSignals(True); w.hide(); w.setParent(None); w.deleteLater()
        self.table.clearContents(); self.table.setRowCount(0)
        self.table.setRowCount(len(layers))
        for i, l in enumerate(self.layers):
            r = len(layers)-1-i
            combo = QComboBox(); combo.addItems(list(presets.MATERIALS.keys()))
            combo.setCurrentText(l.name if l.name in presets.MATERIALS else 'Custom')
            combo.currentTextChanged.connect(lambda name, i=i: self._material(i, name))
            self.table.setCellWidget(r, 0, combo)
            for c, v in ((1, l.n_re), (2, l.n_im)):
                self.table.setItem(r, c, QTableWidgetItem(f'{v:g}'))
            it = QTableWidgetItem('∞' if l.d is None else f'{l.d:g}')
            if l.d is None:
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                it.setToolTip('Semi-infinite medium')
            self.table.setItem(r, 3, it)
            role = 'immersion' if i == 0 else ('exit' if i == len(layers)-1 else f'layer {i}')
            self.table.setVerticalHeaderItem(r, QTableWidgetItem(role))
        self._block = False

    def _material(self, i, name, wavelength=None):
        entry = presets.MATERIALS.get(name)
        self.layers[i].name = name
        if entry:
            fn, note = entry
            n = complex(fn(wavelength or self.wavelength))
            self.layers[i].n_re, self.layers[i].n_im = round(n.real, 6), round(n.imag, 6)
        self.set_layers(self.layers); self.changed.emit()

    wavelength = 0.532

    def refresh_dispersion(self, wavelength):
        self.wavelength = wavelength
        changed = False
        for l in self.layers:
            entry = presets.MATERIALS.get(l.name)
            if entry:
                n = complex(entry[0](wavelength))
                if abs(n - l.n) > 1e-6:
                    l.n_re, l.n_im = round(n.real, 6), round(n.imag, 6); changed = True
        if changed:
            self.set_layers(self.layers)

    def _item_changed(self, item):
        if self._block: return
        r, c = item.row(), item.column(); i = len(self.layers)-1-r
        try:
            v = float(item.text().replace(',', '.'))
        except ValueError:
            self.set_layers(self.layers); return
        l = self.layers[i]
        if c == 1: l.n_re = v
        elif c == 2: l.n_im = v
        elif c == 3 and l.d is not None: l.d = max(0.0, v)
        if c in (1, 2): l.name = 'Custom'
        self.set_layers(self.layers); self.changed.emit()

    def selected_index(self):
        r = self.table.currentRow()
        return None if r < 0 else len(self.layers)-1-r

    def add_layer(self):
        i = self.selected_index()
        if i is None or i >= len(self.layers)-1: i = max(0, len(self.layers)-2)
        new = Layer('Custom', 1.5, 0.0, 0.5)
        if len(self.layers) == 1:
            self.layers.append(Layer('Air', 1.0, 0.0, None))
        else:
            self.layers.insert(i+1, new)
        self.set_layers(self.layers); self.changed.emit()

    def remove_layer(self):
        i = self.selected_index()
        if i is None: i = len(self.layers)-2 if len(self.layers) > 2 else len(self.layers)-1
        if i == 0 or len(self.layers) <= 1: return
        del self.layers[i]
        if len(self.layers) >= 2:
            self.layers[-1].d = None; self.layers[0].d = None
        self.set_layers(self.layers); self.changed.emit()

    def move(self, direction):
        i = self.selected_index()
        if i is None or i in (0, len(self.layers)-1): return
        j = i + direction
        if j <= 0 or j >= len(self.layers)-1: return
        self.layers[i], self.layers[j] = self.layers[j], self.layers[i]
        self.set_layers(self.layers); self.table.selectRow(len(self.layers)-1-j); self.changed.emit()


class AspectPlotFrame(QWidget):
    """Responsive plot frame that never changes the data or PyQtGraph axis ranges.

    The outer grid decides each panel's size; this frame fits the entire plot,
    including labels and colorbar, inside that space without clipping.
    """

    def __init__(self, plot, width_to_height=1.18):
        super().__init__()
        self.plot_widget = plot
        self.ratio = float(width_to_height)
        plot.setParent(self)
        plot.show()
        plot.setMinimumSize(0, 0)
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def resizeEvent(self, event):
        available_width = max(1, self.width())
        available_height = max(1, self.height())
        width = min(available_width, int(available_height * self.ratio))
        height = min(available_height, int(width / self.ratio))
        self.plot_widget.setGeometry(
            (available_width - width) // 2,
            (available_height - height) // 2,
            width,
            height,
        )
        super().resizeEvent(event)


# ======================================================================== main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f'MultiLayerFocus {__version__}')
        if ICON.is_file(): self.setWindowIcon(QIcon(str(ICON)))
        # Use the actual Qt logical screen size (also correct at 125/150% DPI).
        # Keep an ordinary resizable window inside the available desktop area.
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.setMinimumSize(min(640, available.width()), min(430, available.height()))
            self.resize(round(available.width() * 0.97),
                        round(available.height() * 0.95))
        else:
            self.resize(1360, 850)
        self.settings = QSettings('MultiLayerFocus', 'MultiLayerFocus')
        self.theme = self.settings.value('theme', 'light')
        self.scene = Scene()
        self.run_result = None
        self._saved_scene_json = None
        self._last_run_scene_json = None
        self._scene_path = None
        self._dirty = False
        self.worker = None; self.pending = None
        self.sweep_worker = None
        self._loading = False
        self.preview_timer = QTimer(self); self.preview_timer.setSingleShot(True); self.preview_timer.setInterval(220)
        self.preview_timer.timeout.connect(lambda: self.start(preview=True))
        root = QWidget(); root.setObjectName('root'); self.setCentralWidget(root)
        outer = QVBoxLayout(root); outer.setContentsMargins(10, 8, 10, 4); outer.setSpacing(6)
        outer.addWidget(self._header())

        # Single draggable splitter: Parameters 20% / Plots 80% on startup.
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(7)
        self.splitter.addWidget(self._sidebar())
        self.splitter.addWidget(self._center())
        self.splitter.setStretchFactor(0, 2)
        self.splitter.setStretchFactor(1, 8)
        # The width of the parameter panel is a percentage, not fixed pixels.
        outer.addWidget(self.splitter, 1)

        # Calculation summary belongs to the footer only.
        self.status = QStatusBar(); self.setStatusBar(self.status)
        self.status.setMinimumHeight(34)
        self.status_label = QLabel('Ready')
        self.status_label.setStyleSheet('font-size: 13px; padding: 5px 4px;')
        self.status_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.status.addWidget(self.status_label, 1)
        self.hover_label = QLabel(''); self.status.addPermanentWidget(self.hover_label)
        self.apply_theme(self.theme)
        self.set_scene(self.scene)
        self._mark_scene_saved()
        # Never restore the previously selected tab.
        self.tabs.setCurrentIndex(0)
        QTimer.singleShot(0, self._restore_window_layout)

    # ------------------------------------------------------------------ layout
    def _header(self):
        # Original single-row header, with a flexible progress indicator.
        header = QWidget()
        h = QHBoxLayout(header)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        logo = QLabel()
        if ICON.is_file():
            logo.setPixmap(QPixmap(str(ICON)).scaled(
                34, 34, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
        h.addWidget(logo)
        t = QVBoxLayout(); t.setSpacing(0)
        self.brand_title = QLabel('MultiLayerFocus')
        self.brand_title.setObjectName('brand')
        self.brand_subtitle = QLabel('Vectorial focusing through stratified media')
        self.brand_subtitle.setObjectName('brandSub')
        t.addWidget(self.brand_title); t.addWidget(self.brand_subtitle)
        h.addLayout(t)
        h.addStretch(1)

        self.live = QPushButton('● Live preview')
        self.live.setCheckable(True); self.live.setChecked(True)
        self.live.setToolTip('Recompute a fast draft automatically whenever a parameter changes')
        self.live.toggled.connect(lambda on: on and self.schedule_preview())
        h.addWidget(self.live)
        self.compute_btn = QPushButton('Compute')
        self.compute_btn.setObjectName('primary')
        self.compute_btn.setToolTip('Full-quality simulation with automatic accuracy control (Ctrl+R)')
        self.compute_btn.clicked.connect(lambda: self.start(preview=False))
        self.compute_btn.setShortcut('Ctrl+R')
        h.addWidget(self.compute_btn)

        self.progress = QProgressBar()
        self.progress.setMinimumWidth(65)
        self.progress.setMaximumWidth(260)
        self.progress.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.progress.setFixedHeight(24)
        self.progress.setRange(0, 100); self.progress.setValue(0)
        self.progress.setTextVisible(True); self.progress.setFormat('%p%')
        h.addWidget(self.progress, 1)

        exp = QToolButton(); exp.setText('Export'); exp.setObjectName('exportButton')
        exp.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        # Export button styling is controlled by theme.py for both themes.
        menu = QMenu(self)
        for label, slot in (
            ('Figure of Current Tab (PNG)...', self.export_view_png),
            ('Current Tab Data (TXT)...', self.export_tab_txt),
            ('Current Tab Data (MAT)...', self.export_tab_mat),
        ):
            action = menu.addAction(label)
            action.triggered.connect(slot)
        menu.addSeparator()
        action = menu.addAction('Export All Tabs (TXT + MAT)...')
        action.triggered.connect(self.export_all_tabs)
        menu.addSeparator()
        for label, slot in (
            ('Save scene (JSON)...', self.save_scene),
            ('Open scene (JSON)...', self.open_scene),
        ):
            action = menu.addAction(label)
            action.triggered.connect(slot)
        menu.addSeparator()
        action = menu.addAction('Detailed simulation report (PDF)...')
        action.triggered.connect(self.export_report)
        exp.setMenu(menu); h.addWidget(exp)

        # A real painted icon remains legible even when Windows substitutes fonts.
        self.theme_btn = QPushButton()
        self.theme_btn.setToolTip('Light / dark theme')
        self.theme_btn.setFixedWidth(40)
        self.theme_btn.setIcon(_theme_toggle_icon(self.theme))
        self.theme_btn.setIconSize(QSize(26, 26))
        self.theme_btn.clicked.connect(lambda: self.apply_theme('dark' if self.theme == 'light' else 'light'))
        h.addWidget(self.theme_btn)
        about = QPushButton('?')
        about.setFixedWidth(40); about.clicked.connect(self.about)
        h.addWidget(about)
        return header

    def _sidebar(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(200)
        scroll.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        # No maximum width: its size is determined by the draggable splitter.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.sidebar_scroll = scroll
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(0, 0, 4, 0); lay.setSpacing(7)
        scroll.setWidget(w)
        # Objective
        c = Card('Objective', 'Configure the numerical aperture, focal length, and objective parameters.')
        self.obj_combo = QComboBox(); self.obj_combo.addItems([o['name'] for o in presets.OBJECTIVES] + ['Custom'])
        self.obj_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.obj_combo.setMinimumContentsLength(14)
        self.obj_combo.setToolTip(self.obj_combo.currentText())
        self.obj_combo.currentTextChanged.connect(self.obj_combo.setToolTip)
        self.obj_combo.activated.connect(self.objective_chosen)
        c.add(self.obj_combo)
        self.na = SliderSpin('Numerical aperture', 0.95, 0.05, 1.65, 3, step=0.01); c.add(self.na)
        self.focal = Spin(2620, 1, 1e6, 1, 10, ' [µm]'); self.focal.setFixedWidth(105)
        c.add(row('Focal length', self.focal, 'Objective focal length = tube-lens focal length / magnification'))
        self.n_note = QLabel(''); self.n_note.setObjectName('muted'); self.n_note.setWordWrap(True); c.add(self.n_note)
        lay.addWidget(c)
        self.na.valueChanged.connect(self.objective_modified)
        self.focal.valueChanged.connect(self.objective_modified)
        # Beam
        c = Card('Illumination', 'Define the incident beam amplitude, polarization, phase, and aberrations.')
        self.wl = SliderSpin('Vacuum wavelength', 0.532, 0.23, 1.0, 4, 'µm', step=0.001); c.add(self.wl)

        wl_base_style = self.wl.spin.styleSheet()

        def update_wavelength_color(wl):
            style = wl_base_style

            if 0.380 <= wl <= 0.750:
                points = [
                    (0.380, 0.77), (0.450, 0.66),
                    (0.490, 0.50), (0.530, 0.33),
                    (0.580, 0.17), (0.620, 0.08),
                    (0.750, 0.00)
                ]
                for (a, ha), (b, hb) in zip(points, points[1:]):
                    if a <= wl <= b:
                        hue = ha + (hb - ha) * (wl - a) / (b - a)
                        color = QColor.fromHsvF(hue, 0.85, 0.70).name()
                        style += f"\nQDoubleSpinBox {{ color: {color}; }}"
                        break

            self.wl.spin.setStyleSheet(style)

        self.wl.valueChanged.connect(update_wavelength_color)
        update_wavelength_color(self.wl.value())

        self.pol = QComboBox()
        for label, code in [
            ('Linear-X', 'linear-x'),
            ('Linear-Y', 'linear-y'),
            ('Linear-45°', 'linear-45°'),
            ('Circular-L', 'circular-L'),
            ('Circular-R', 'circular-R'),
            ('Radial', 'radial'),
            ('Azimuthal', 'azimuthal'),
            ('Custom (Elliptical)', 'custom (elliptical)'),
            ('Custom (Cylindrical)', 'custom (cylindrical)')
        ]:
            self.pol.addItem(label, code)
        c.add(row('Polarization', self.pol))
        self.gamma = Spin(0, 0, 90, 1, 1, ' [°]'); self.beta = Spin(0, -180, 180, 1, 1, ' [°]'); self.delta = Spin(0, -90, 90, 1, 1, ' [°]')
        for wdg in (self.gamma, self.beta, self.delta): wdg.setFixedWidth(88)
        self.custom_pol = QWidget(); cp = QVBoxLayout(self.custom_pol); cp.setContentsMargins(0, 0, 0, 0)
        cp.addWidget(row('Amplitude angle γ', self.gamma, 'Ex ∝ cos γ, Ey ∝ sin γ'))
        cp.addWidget(row('Relative phase β', self.beta, 'Phase of Ey relative to Ex'))
        self.custom_cyl = QWidget(); cc = QVBoxLayout(self.custom_cyl); cc.setContentsMargins(0, 0, 0, 0)
        cc.addWidget(row('Radial→azimuthal δ', self.delta, '0 = radial, 90 [°] = azimuthal'))
        c.add(self.custom_pol); c.add(self.custom_cyl)
        self.vortex = ISpin(0, -20, 20); self.vortex.setFixedWidth(100)
        c.add(row('Vortex charge ℓ', self.vortex, 'Helical phase exp(iℓϕ) in the pupil'))
        self.uniform = QCheckBox('Uniform pupil (overfilled)'); self.uniform.setChecked(True); c.add(self.uniform)
        self.fill = SliderSpin('Gaussian fill  w₀ / ρmax', 1.0, 0.1, 4.0, 2, step=0.05); c.add(self.fill)
        self.power = Spin(1.0, 0.0001, 1e6, 4, 0.1, 'mW'); self.power.setFixedWidth(105)
        c.add(row('Power in the aperture', self.power, 'Used for absolute units [V/m], [W/cm²], and [W/cm³]'))
        zr = QHBoxLayout(); self.zern_label = QLabel('Aberrations: none'); zr.addWidget(self.zern_label); zr.addStretch(1)
        zb = QPushButton('Edit…'); zb.clicked.connect(self.edit_zernike); zr.addWidget(zb); c.add_layout(zr)
        pr = QHBoxLayout(); self.pupil_label = QLabel('Pupil: generated'); self.pupil_label.setWordWrap(True)
        pr.addWidget(self.pupil_label, 1)
        pb = QPushButton('Import…'); pb.setToolTip('Measured or designed complex pupil: four text files Ex, Ey (rows = y, '
                                                     'columns = x; complex numbers like (1+2j)), x and y coordinate vectors in [µm]')
        pb.clicked.connect(self.import_pupil); pr.addWidget(pb)
        pc = QPushButton('×'); pc.setFixedWidth(30); pc.setToolTip('Back to the generated pupil'); pc.clicked.connect(self.clear_pupil)
        pr.addWidget(pc); c.add_layout(pr)
        lay.addWidget(c)
        # Stack
        c = Card('Multilayer', 'Configure the refractive indices, thicknesses, and geometry of the optical stack.', expanded=False)
        self.stack = StackEditor(); c.add(self.stack)
        self.zout = Spin(0, -1e5, 1e5, 4, 0.01, 'µm'); self.zout.setFixedWidth(105)
        c.add(row('Last interface z_out', self.zout, 'Position of the last interface relative to the nominal focus z = 0'))
        lay.addWidget(c)

        # View
        c = Card('Observation', 'Set the spatial observation region and sampling resolution.', expanded=False)
        self.z0 = SliderSpin('XY plane z₀', 0.0, -10, 10, 4, 'µm', step=0.01); c.add(self.z0)
        g = QGridLayout(); g.setHorizontalSpacing(8)
        self.lateral = Spin(2.0, 0.05, 500, 3, 0.1, 'µm'); self.axial = Spin(3.0, 0.05, 1000, 3, 0.1, 'µm')
        self.x0 = Spin(0, -1e3, 1e3, 3, 0.05, 'µm'); self.y0 = Spin(0, -1e3, 1e3, 3, 0.05, 'µm')
        self.nxy = ISpin(121, 11, 1001); self.nz = ISpin(121, 11, 1001)
        # Two columns fit a 20%-width sidebar without truncating input boxes.
        for i, (lab, wdg) in enumerate((('Lateral size', self.lateral), ('Axial size', self.axial), ('Center x₀', self.x0),
                                        ('Center y₀', self.y0), ('Samples xy', self.nxy), ('Samples z', self.nz))):
            g.addWidget(QLabel(lab), i, 0)
            g.addWidget(wdg, i, 1)
        g.setColumnStretch(0, 1)
        c.add_layout(g)
        lay.addWidget(c)
        # Accuracy
        c = Card('Accuracy', 'Configure numerical accuracy and optional 3D volume reconstruction.', expanded=False)
        self.quality = QComboBox(); self.quality.addItems(['auto', 'draft', 'standard', 'high', 'custom'])
        self.quality.setToolTip('draft/standard/high: Gauss–Legendre nodes chosen adaptively until a refined grid agrees '
                                'to 3e-2 / 1e-3 / 1e-5; auto = high for computed results (the live preview uses draft); '
                                'custom: fixed node counts')
        c.add(row('Angular sampling', self.quality))
        self.ntheta = ISpin(64, 4, 4000); self.nphi = ISpin(96, 5, 6000)
        self.custom_q = QWidget(); cq = QHBoxLayout(self.custom_q); cq.setContentsMargins(0, 0, 0, 0)
        cq.addWidget(QLabel('Nθ')); cq.addWidget(self.ntheta); cq.addWidget(QLabel('Nϕ')); cq.addWidget(self.nphi)
        c.add(self.custom_q)
        self.do_volume = QCheckBox('Compute full volume'); c.add(self.do_volume)
        self.vxy = ISpin(61, 10, 301); self.vz = ISpin(61, 10, 301)
        vr = QVBoxLayout()
        vr.setSpacing(4)
        vr.addWidget(row('Volume samples XY', self.vxy))
        vr.addWidget(row('Volume samples Z', self.vz))
        c.add_layout(vr)
        lay.addWidget(c)
        lay.addStretch(1)
        # connections
        for wdg in (self.na, self.wl, self.fill, self.z0):
            wdg.valueChanged.connect(self.changed)
        for wdg in (self.focal, self.gamma, self.beta, self.delta, self.power, self.zout, self.lateral, self.axial, self.x0, self.y0):
            wdg.valueChanged.connect(self.changed)
        for wdg in (self.vortex, self.nxy, self.nz, self.ntheta, self.nphi, self.vxy, self.vz):
            wdg.valueChanged.connect(self.changed)
        self.pol.currentTextChanged.connect(self.changed); self.quality.currentTextChanged.connect(self.changed)
        self.uniform.toggled.connect(self.changed); self.do_volume.toggled.connect(self.changed)
        self.stack.changed.connect(self.changed)
        return scroll


    def _install_tab_descriptions(self):
        descriptions = {
            'pupil plane':
                'Complex pupil-field components: intensity and phase '
                'of Ex and Ey across the objective entrance pupil.',

            'pupil':
                'Complex pupil-field components: intensity and phase '
                'of Ex and Ey across the objective entrance pupil.',

            'stack response':
                'Angle-dependent reflection, transmission and absorption, '
                'Fresnel amplitude coefficients, and the optical stack.',

            'focal field':
                'Vectorial focal-field distribution in XY and XZ/YZ. '
                'Use the sliders to select the positions of the '
                'transverse and axial intensity profiles.',

            'focal field components':
                'Cartesian electric-field components in the XY plane: '
                'individual intensities and phases.',

            'polarization':
                'Local three-dimensional polarization ellipses, '
                'ellipticity and spin density. Use Threshold to restrict '
                'ellipse overlays to significant field intensity. '
                'A, B and N can be inspected separately.',

             'energy':
                'Energy transport and absorption in the multilayer system. '
                'The maps show the axial Poynting flux and absorbed power '
                'density, while the energy budget summarizes reflection, '
                'transmission and absorption for the incident beam.',

            'full volume reconstruction':
                'Three-dimensional field reconstruction: an XY slice, '
                'a maximum-intensity XY projection and a selectable '
                'XZ/YZ projection. Enable \"Compute full volume\" in the Accuracy panel, '
                'then press Compute. The full 3D viewer opens separately.',

            'parameter sweep':
                'Vary one simulation parameter and examine the '
                'corresponding numerical outcome across multiple runs.',

            'sweep':
                'Vary one simulation parameter and examine the '
                'corresponding numerical outcome across multiple runs.'
        }

        for i in range(self.tabs.count()):
            title = self.tabs.tabText(i)
            description = descriptions.get(title.strip().lower())

            if not description:
                continue

            original_widget = self.tabs.widget(i)
            self.tabs.removeTab(i)

            container = QWidget()
            layout = QVBoxLayout(container)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(6)

            note = QLabel(description)
            note.setObjectName('muted')
            note.setWordWrap(True)

            layout.addWidget(note)
            original_widget.setMinimumSize(0, 0)
            original_widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
            layout.addWidget(original_widget, 1)

            self.tabs.insertTab(i, container, title)

            # Restore the original tab content after reparenting
            original_widget.show()




    def _center(self):
        self.tabs = QTabWidget(); self.tabs.setDocumentMode(False)
        self.tabs.setMinimumSize(0, 0)
        self.tabs.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.tabs.tabBar().setUsesScrollButtons(True)
        self.tabs.addTab(self._pupil_tab(), 'Pupil Plane')
        self.tabs.addTab(self._response_tab(), 'Stack Response')
        self.tabs.addTab(self._focus_tab(), 'Focal Field')
        self.tabs.addTab(self._components_tab(), 'Focal Field Components')
        self.tabs.addTab(self._polarization_tab(), 'Polarization')
        self.tabs.addTab(self._energy_tab(), 'Energy')
        self.tabs.addTab(self._volume_tab(), 'Full Volume Reconstruction')
        self.tabs.addTab(self._sweep_tab(), 'Parameter Sweep')
        self._install_tab_descriptions()
        self.tabs.currentChanged.connect(lambda _: self.render())
        return self.tabs

    def _horizontal_strip(self, content_layout):
        """Single-row controls, sized by their layout; never clip in a scroll area."""
        host = QWidget()
        content_layout.setContentsMargins(5, 6, 5, 6)
        content_layout.setSpacing(6)
        host.setLayout(content_layout)
        host.setMinimumHeight(40)
        host.setMinimumWidth(0)
        host.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        return host

    def _focus_tab(self):
        w = QWidget(); l = QVBoxLayout(w); l.setContentsMargins(10, 10, 10, 10); l.setSpacing(8)
        tiles = QHBoxLayout(); tiles.setSpacing(6)
        self.t_peak = MetricTile('Peak |E| [V/m]', tooltip='Maximum field amplitude in the XZ section for the chosen power')
        self.t_int = MetricTile('Peak intensity [W/cm²]', tooltip='n|E|²/2Z₀ at the maximum')
        self.t_zpk = MetricTile('Peak position z [µm]', tooltip='Axial position of the intensity maximum (focal shift)')
        self.t_fw = MetricTile('FWHM x · y · z [nm]', tooltip='Full widths at half maximum of |E|²')
        self.t_ez = MetricTile('|Ez|² share in XY', tooltip='Integrated fraction of the longitudinal component in the XY plane')
        self.t_rta = MetricTile('R · T · A of the beam', tooltip='Exact beam-level reflectance, transmittance and absorptance')
        self.t_acc = MetricTile('accuracy', tooltip='Estimated relative angular-sampling error (full runs)')
        # Keep all metrics on their original single row; the Energy and
        # Polarization tabs already expose R/T/A and the Ez contribution.
        for t in (self.t_peak, self.t_int, self.t_zpk, self.t_fw, self.t_acc):
            tiles.addWidget(t)
        l.addLayout(tiles)
        ctr = QHBoxLayout(); ctr.setSpacing(10)
        def add_pair(caption, widget):
            pair = QHBoxLayout()
            pair.setContentsMargins(0, 0, 0, 0)
            pair.setSpacing(4)
            pair.addWidget(QLabel(caption))
            pair.addWidget(widget)
            ctr.addLayout(pair)
        self.qty = QComboBox()
        self.qty.addItems(
            [q for q in QUANTITIES if not q.startswith('arg ')]
        )
        self.qty.setMinimumWidth(100)
        self.qty.setMaximumWidth(155)
        add_pair('Show', self.qty)
        # Axial section selector
        self.focus_axial_plane = QComboBox()
        self.focus_axial_plane.addItems(['XZ section', 'YZ section'])
        self.focus_axial_plane.setMaximumWidth(115)

        add_pair('Axial section', self.focus_axial_plane)

        self.focus_axial_plane.currentIndexChanged.connect(
            lambda _: self.render()
        )
        self.cmap = QComboBox(); self.cmap.addItems(CMAPS)
        self.cmap.setMaximumWidth(96)
        add_pair('Color Map', self.cmap)
        self.logscale = QCheckBox('Log scale'); ctr.addWidget(self.logscale)
        self.norm = QComboBox(); self.norm.addItems(['normalized', 'absolute units'])
        self.norm.setMaximumWidth(120)
        ctr.addWidget(self.norm)
        self.shared = QCheckBox('Same scale: axial/XY'); self.shared.setChecked(False); ctr.addWidget(self.shared)
        ctr.addStretch(1)
        for wdg in (self.qty, self.cmap, self.norm):
            wdg.currentTextChanged.connect(lambda _: self.render())
        self.logscale.toggled.connect(lambda _: self.render()); self.shared.toggled.connect(lambda _: self.render())
        l.addWidget(self._horizontal_strip(ctr))

        # A shared grid gives both image frames the same height and width.
        # The two original side-by-side maps and profiles are retained.
        columns = QGridLayout()
        columns.setHorizontalSpacing(8)
        columns.setVerticalSpacing(4)

        self.map_xz = MapView('x [µm]', 'z [µm]')
        self.map_xz.plot.setAspectLocked(False)
        self.map_xy = MapView('x [µm]', 'y [µm]')
        columns.addWidget(AspectPlotFrame(self.map_xz, 1.18), 0, 0)
        columns.addWidget(AspectPlotFrame(self.map_xy, 1.18), 0, 1)
        columns.setColumnStretch(0, 1)
        columns.setColumnStretch(1, 1)

        # Keep the existing coordinate readout on mouse hover
        for m in (self.map_xz, self.map_xy):
            m.hovered.connect(
                lambda x, y, v: self.hover_label.setText(
                    f'({x:.3f}, {y:.3f}) [µm]   value {v:.4g}'
                )
            )

        # Position controls
        self.focus_axial_slider = QSlider(Qt.Orientation.Horizontal)
        self.focus_axial_z_slider = QSlider(Qt.Orientation.Horizontal)        
        self.focus_xy_x_slider = QSlider(Qt.Orientation.Horizontal)
        self.focus_xy_y_slider = QSlider(Qt.Orientation.Horizontal)

        for slider in (self.focus_axial_slider, self.focus_axial_z_slider, self.focus_xy_x_slider, self.focus_xy_y_slider):
            slider.setRange(0, 0)
            slider.setSingleStep(1)
            slider.setMaximumWidth(255)

        self.focus_axial_label = QLabel('X = 0.000 [µm]')
        self.focus_axial_z_label = QLabel('Z = 0.000 [µm]')        
        self.focus_xy_x_label = QLabel('X = 0.000 [µm]')
        self.focus_xy_y_label = QLabel('Y = 0.000 [µm]')

        # Slider below the axial image
        ax_row = QHBoxLayout()
        ax_row.addWidget(QLabel('Z profile at'))
        ax_row.addWidget(self.focus_axial_slider, 1)
        ax_row.addWidget(self.focus_axial_label)
        columns.addLayout(ax_row, 1, 0)
        az_row = QHBoxLayout()
        az_row.addWidget(QLabel('Z position'))
        az_row.addWidget(self.focus_axial_z_slider, 1)
        az_row.addWidget(self.focus_axial_z_label)
        columns.addLayout(az_row, 2, 0)        

        # Two sliders below the XY image
        x_row = QHBoxLayout()
        x_row.addWidget(QLabel('X position'))
        x_row.addWidget(self.focus_xy_x_slider, 1)
        x_row.addWidget(self.focus_xy_x_label)
        columns.addLayout(x_row, 1, 1)

        y_row = QHBoxLayout()
        y_row.addWidget(QLabel('Y position'))
        y_row.addWidget(self.focus_xy_y_slider, 1)
        y_row.addWidget(self.focus_xy_y_label)
        columns.addLayout(y_row, 2, 1)

        # Profile plots: directly below their corresponding images
        self.prof_ax = LinePlot('z [µm]', 'normalized')
        self.prof_lat = LinePlot('position [µm]', 'normalized')

        self.prof_ax.legend.hide()
        self.prof_lat.legend.hide()
        self.prof_ax.setMinimumSize(0, 0)
        self.prof_lat.setMinimumSize(0, 0)
        columns.addWidget(self.prof_ax, 3, 0)
        columns.addWidget(self.prof_lat, 3, 1)
        columns.setRowStretch(0, 6)
        columns.setRowStretch(3, 2)

        # Visible extraction lines
        colors = PALETTES[self.theme]

        self.focus_axial_vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(colors['accent2'], width=1.5))
        self.focus_axial_hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(colors['accent'], width=1.5))


        # Vertical line: fixed X, corresponding to the Y profile
        self.focus_xy_vline = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(colors['accent2'], width=1.5)
        )

        # Horizontal line: fixed Y, corresponding to the X profile
        self.focus_xy_hline = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=pg.mkPen(colors['accent'], width=1.5)
)

        self.map_xz.plot.addItem(
            self.focus_axial_vline, ignoreBounds=True
        )
        self.map_xz.plot.addItem(self.focus_axial_hline, ignoreBounds=True)        
        self.map_xy.plot.addItem(
            self.focus_xy_vline, ignoreBounds=True
        )
        self.map_xy.plot.addItem(
            self.focus_xy_hline, ignoreBounds=True
        )

        # Dynamic profile updates without recomputing the field
        self.focus_axial_slider.valueChanged.connect(
            self._focus_update_profiles
        )
        self.focus_axial_z_slider.valueChanged.connect(self._focus_update_profiles)        
        self.focus_xy_x_slider.valueChanged.connect(
            self._focus_update_profiles
        )
        self.focus_xy_y_slider.valueChanged.connect(
            self._focus_update_profiles
        )

        l.addLayout(columns, 1)

        return w



    def _components_tab(self):
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(3, 3, 3, 3)
        l.setSpacing(2)

        # Original 2 x 3 grid, identical responsive plot frames.
        g = QGridLayout()
        g.setHorizontalSpacing(2)
        g.setVerticalSpacing(2)
        self.comp_maps = []

        for i in range(6):
            m = MapView('x [µm]', 'y [µm]')
            g.addWidget(AspectPlotFrame(m, 1.18), i // 3, i % 3)
            self.comp_maps.append(m)
        for col in range(3):
            g.setColumnStretch(col, 1)
        for r_i in range(2):
            g.setRowStretch(r_i, 1)

        l.addLayout(g, 1)

        # Keep the scale control above the plots, aligned to the top right.
        self.comp_shared = QCheckBox('Shared intensity scale')
        self.comp_shared.setChecked(True)
        self.comp_shared.toggled.connect(lambda _: self.render())
        top_controls = QHBoxLayout()
        top_controls.setContentsMargins(0, 0, 2, 0)
        top_controls.addStretch(1)
        top_controls.addWidget(self.comp_shared)
        l.insertLayout(0, top_controls)

        return w


    def _polarization_tab(self):
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(6, 6, 6, 6)
        l.setSpacing(5)

        controls = QHBoxLayout()
        controls.setSpacing(6)

        self.pol_ellipse_density = QSlider(Qt.Orientation.Horizontal)
        self.pol_ellipse_density.setRange(5, 30)
        self.pol_ellipse_density.setValue(15)
        self.pol_ellipse_density.setMaximumWidth(130)
        self.pol_density_label = QLabel('15 per axis')

        self.pol_ellipse_size = QSlider(Qt.Orientation.Horizontal)
        self.pol_ellipse_size.setRange(10, 90)
        self.pol_ellipse_size.setValue(45)
        self.pol_ellipse_size.setMaximumWidth(125)
        self.pol_size_label = QLabel('45%')

        # All polarization maps and ellipse overlays use this display mask.
        self.pol_ellipse_threshold = QSlider(Qt.Orientation.Horizontal)
        self.pol_ellipse_threshold.setRange(0, 100)
        self.pol_ellipse_threshold.setValue(0)
        self.pol_ellipse_threshold.setMaximumWidth(130)
        self.pol_threshold_label = QLabel('0%')
        self.pol_ellipse_threshold.setToolTip(
            'Hide polarization data where total field intensity '
            'is below this percentage of the XY-plane maximum.'
                )

        for caption, slider, value_label in (
            ('Ellipse density', self.pol_ellipse_density, self.pol_density_label),
            ('Ellipse size', self.pol_ellipse_size, self.pol_size_label),
            ('Threshold', self.pol_ellipse_threshold, self.pol_threshold_label),
        ):
            controls.addWidget(QLabel(caption))
            controls.addWidget(slider)
            controls.addWidget(value_label)
            controls.addSpacing(7)

        controls.addStretch(1)
        self.pol_abn_button = QPushButton('Show A, B, N components...')
        self.pol_abn_button.setObjectName('primary')
        self.pol_abn_button.clicked.connect(self._show_abn_dialog)
        controls.addWidget(self.pol_abn_button)
        l.addWidget(self._horizontal_strip(controls))

        # Left: one large square field map. Right: two equal square maps.
        # The scale of the physical XY axes remains 1:1 everywhere.
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(4)
        self.ell_map = MapView('x [µm]', 'y [µm]')
        self.pol_maps = [MapView('x [µm]', 'y [µm]') for _ in range(2)]
        body.addWidget(AspectPlotFrame(self.ell_map, 1.18), 3)

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(4)
        for m in self.pol_maps:
            right.addWidget(AspectPlotFrame(m, 1.18), 1)
        body.addLayout(right, 2)

        for m in (self.ell_map, *self.pol_maps):
            m.plot.setAspectLocked(True, ratio=1)
            m.plot.getAxis('left').setWidth(47)
            m.plot.getAxis('bottom').setHeight(41)
            m.bar.axis.setWidth(40)
        l.addLayout(body, 1)

        self.ellipse_items = []
        self._pol_result = None
        self._pol_ellipse = None
        self.pol_abn_dialog = None

        self.pol_ellipse_density.valueChanged.connect(self._redraw_pol_ellipses)
        self.pol_ellipse_size.valueChanged.connect(self._redraw_pol_ellipses)
        self.pol_ellipse_threshold.valueChanged.connect(self._polarization_threshold_changed)
        return w

    def _polarization_threshold_changed(self, *_):
        """Apply the intensity threshold to every polarization map."""
        self.pol_threshold_label.setText(
            f'{self.pol_ellipse_threshold.value()}%'
        )

        if self._pol_result is not None:
            self.render_polarization(self._pol_result)

    def _energy_tab(self):
        w = QWidget(); l = QVBoxLayout(w); l.setContentsMargins(5, 5, 5, 5)
        l.setSpacing(4)

        # Matching axial maps with the same height/width ratio as the
        # tall axial projection in Full volume reconstruction.
        maps = QGridLayout()
        maps.setHorizontalSpacing(12)
        maps.setVerticalSpacing(2)
        self.e_sxz = MapView('x [µm]', 'z [µm]')
        self.e_abs = MapView('x [µm]', 'z [µm]')
        for col, m in enumerate((self.e_sxz, self.e_abs)):
            m.plot.setAspectLocked(False)
            maps.addWidget(AspectPlotFrame(m, 0.70), 0, col)
            maps.setColumnStretch(col, 1)
        maps.setRowStretch(0, 1)
        l.addLayout(maps, 3)

        self.budget_plot = pg.PlotWidget(viewBox=UnifiedViewBox())
        setup_unified_plot(self.budget_plot.getPlotItem()); self.budget_plot.hideButtons(); self.budget_plot.setMaximumHeight(170)
        self.budget_plot.setMouseEnabled(False, False)
        l.addWidget(self.budget_plot, 1)
        return w


    def _response_tab(self):
        w = QWidget(); l = QVBoxLayout(w); l.setContentsMargins(10, 10, 10, 10)
        h = QHBoxLayout()
        self.resp_R = LinePlot('incidence angle in the immersion medium [°]', 'power fraction')
        self.resp_amp = LinePlot('incidence angle in the immersion medium [°]', '|r|, |t|')
        # Explicit titles and axis labels for the stack response
        fg = PALETTES[self.theme]['plot_fg']

        self.resp_R.setTitle(
            'Reflection, transmission and absorption',
            color=fg
        )
        self.resp_R.setLabel('bottom', 'Incidence angle [°]')
        self.resp_R.setLabel('left', 'Power fraction')

        self.resp_amp.setTitle(
            'Fresnel amplitude coefficients',
            color=fg
        )
        self.resp_amp.setLabel('bottom', 'Incidence angle [°]')
        self.resp_amp.setLabel('left', 'Amplitude |r|, |t|')

        for plot in (self.resp_R, self.resp_amp):
            plot.showAxis('bottom', True)
            plot.showAxis('left', True)


        h.setSpacing(18)
        # External legends: the curves are never obscured by legend panels.
        self.resp_R.legend.hide()
        self.resp_amp.legend.hide()
        self.response_legend_labels = []
        for plot, legends in (
            (self.resp_R, ('R s', 'R p', 'T s', 'T p', 'A s', 'A p')),
            (self.resp_amp, ('|r s|', '|r p|', '|t s|', '|t p|')),
        ):
            block = QHBoxLayout()
            block.setSpacing(3)
            block.addWidget(plot, 1)
            legends_host = QWidget()
            legends_layout = QVBoxLayout(legends_host)
            legends_layout.setContentsMargins(0, 24, 0, 0)
            legends_layout.setSpacing(6)
            for label in legends:
                legend_label = QLabel()
                legend_label.setTextFormat(Qt.TextFormat.RichText)
                self.response_legend_labels.append((legend_label, label))
                legends_layout.addWidget(legend_label)
            legends_layout.addStretch(1)
            legends_host.setFixedWidth(73)
            block.addWidget(legends_host)
            h.addLayout(block, 1)
        self._update_response_legends()
        l.addLayout(h, 3)
        self.big_diagram = StackDiagram(); l.addWidget(self.big_diagram, 2)
        return w


    def _pupil_tab(self):
        w = QWidget()
        layout = QGridLayout(w)
        layout.setContentsMargins(6, 5, 6, 6)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(7)

        # Four interactive PyQtGraph pupil maps
        self.pup_I = MapView('x [mm]', 'y [mm]')
        self.pup_Iy = MapView('x [mm]', 'y [mm]')
        self.pup_P = MapView('x [mm]', 'y [mm]')
        self.pup_Py = MapView('x [mm]', 'y [mm]')

        layout.addWidget(AspectPlotFrame(self.pup_I, 1.18), 0, 0)
        layout.addWidget(AspectPlotFrame(self.pup_Iy, 1.18), 0, 1)
        layout.addWidget(AspectPlotFrame(self.pup_P, 1.18), 1, 0)
        layout.addWidget(AspectPlotFrame(self.pup_Py, 1.18), 1, 1)
        for row_idx in range(2):
            layout.setRowStretch(row_idx, 1)
            layout.setColumnStretch(row_idx, 1)

        return w



    def _volume_tab(self):
        w = QWidget()

        l = QVBoxLayout(w)
        l.setContentsMargins(6, 6, 6, 6)
        l.setSpacing(6)

        # Top controls
        bar = QHBoxLayout()
        bar.setSpacing(7)

        bar.addWidget(QLabel('Slice z'))

        self.vol_slider = SliderSpin(
            '', 0.0, -1, 1, 3, 'µm', compact=True
        )
        self.vol_slider.setMinimumWidth(175)
        self.vol_slider.setMaximumWidth(340)

        bar.addWidget(self.vol_slider, 1)
        self.vol_normalize = QCheckBox('Normalize')
        self.vol_normalize.setChecked(True)
        self.vol_normalize.setToolTip('Normalize to the global volume maximum')
        self.vol_normalize.toggled.connect(lambda _: self.render())
        bar.addWidget(self.vol_normalize)
        # Select the axial projection
        bar.addWidget(QLabel('Axial section'))

        self.vol_axial_plane = QComboBox()
        self.vol_axial_plane.addItems(['XZ section', 'YZ section'])

        self.vol_axial_plane.setMaximumWidth(120)
        bar.addWidget(self.vol_axial_plane)

        # Switch views without recomputing the volume
        self.vol_axial_plane.currentIndexChanged.connect(
            lambda _: self.render()
)

        self.vtk_btn = QPushButton('Open full 3D view...')
        self.vtk_btn.setToolTip(
            'Open the VTK scientific 3D volume viewer'
        )
        self.vtk_btn.clicked.connect(self.open_cpu_volume)

        bar.addWidget(self.vtk_btn)
        bar.addStretch(1)
        l.addWidget(self._horizontal_strip(bar))

        # Keep the original map indices and their meanings
        # Three maps: two XY views and one selectable axial view
        self.vol_maps = [
            MapView('x [µm]', 'y [µm]'),  # XY slice
            MapView('x [µm]', 'y [µm]'),  # XY maximum projection
            MapView('x [µm]', 'z [µm]')   # XZ or YZ projection
        ]

        # Preserve physical axis proportions
        # Square XY views
        for m in self.vol_maps[:2]:
            m.plot.setAspectLocked(True, ratio=1)

        # Tall axial view
        self.vol_maps[2].volume_axial_stretch = True
        self.vol_maps[2].plot.setAspectLocked(False)


        # LEFT: two tall axial projections
        axial_layout = QHBoxLayout()
        axial_layout.setContentsMargins(0, 0, 0, 0)
        axial_layout.setSpacing(4)

        # LEFT: one tall axial projection, selectable XZ / YZ
        axial_layout.addWidget(
            AspectPlotFrame(
                self.vol_maps[2],
                width_to_height=0.70
            ),
            1
        )

        axial_host = QWidget()
        axial_host.setLayout(axial_layout)

        # RIGHT: two square XY maps
        xy_layout = QVBoxLayout()
        xy_layout.setContentsMargins(0, 0, 0, 0)
        xy_layout.setSpacing(4)

        xy_layout.addWidget(
            AspectPlotFrame(
                self.vol_maps[0],
                width_to_height=1.18
            ),
            1
        )

        xy_layout.addWidget(
            AspectPlotFrame(
                self.vol_maps[1],
                width_to_height=1.18
            ),
            1
        )

        xy_host = QWidget()
        xy_host.setLayout(xy_layout)

        # Equal left and right sections
        body = QHBoxLayout()
        body.setSpacing(3)

        body.addWidget(axial_host, 1)
        body.addWidget(xy_host, 1)

        l.addLayout(body, 1)

        # Slice updates
        self.vol_slider.valueChanged.connect(
            lambda _: self.render()
        )

        self.volume_panel = None
        self.vtk_window = None

        return w



    def open_cpu_volume(self):
        """Open the existing VTK volume renderer with its updated Qt layout."""
        self.open_vtk()

    def open_vtk(self):
        r = self.run_result

        if r is None or r.volume is None:
            QMessageBox.information(
                self,
                '3D renderer',
                'Compute a volume first.'
            )
            return

        try:
            from .vtk_viewer_gui import ModernVolumePanel

            # Recreate the VTK window to avoid stale rendering contexts
            if self.vtk_window is not None:
                self.vtk_window.hide()
                self.volume_panel.shutdown()
                self.vtk_window.close()
                self.vtk_window.deleteLater()
                self.vtk_window = None
                self.volume_panel = None
            if self.vtk_window is None:
                dialog = QDialog(self)
                dialog.setWindowTitle(
                    'MultiLayerFocus · Volume viewer'
                )
                screen = QGuiApplication.primaryScreen()
                if screen is not None:
                    available = screen.availableGeometry()
                    dialog.resize(
                        min(1200, int(available.width() * 0.92)),
                        min(780, int(available.height() * 0.92)),
                    )
                else:
                    dialog.resize(1200, 780)

                layout = QVBoxLayout(dialog)

                panel = ModernVolumePanel()
                # The viewer uses the older DecimalSpinBox / PercentageSpinBox.
                # Keep their native text painting (NoButtons made the values
                # disappear on some Windows Qt styles). Hide only the arrows
                # using the same QSS approach as the working main GUI Spin.
                for spin in panel.findChildren(QAbstractSpinBox):
                    spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
                    spin.setStyleSheet('''
                        QDoubleSpinBox::up-button,
                        QDoubleSpinBox::down-button {
                            width: 0px;
                            border: none;
                        }
                    ''')
                layout.addWidget(panel)

                self.vtk_window = dialog
                self.volume_panel = panel

            # Adapt the existing scientific result for the viewer
            adapter = SimpleNamespace(
                volume=r.volume,
                volume_x=r.vx,
                volume_y=r.vy,
                volume_z=r.vz,
                config=r.config
            )

            self.vtk_window.show()
            self.volume_panel.render('3D', adapter)
            # Match the current scientific colormap and axis typography.
            panel = self.volume_panel
            if hasattr(panel, 'set_palette'):
                panel.set_palette(self.cmap.currentText())
            if getattr(panel, 'slice_colorbar', None) is not None:
                panel.slice_colorbar.set_label('Normalized intensity')
                panel.slice_colorbar.ax.tick_params(labelsize=9)
            if getattr(panel, 'slice_artist', None) is not None:
                ax = panel.slice_artist.axes
                ax.tick_params(labelsize=9, direction='in')
                ax.set_xlabel('x [µm]', fontsize=10)
                ax.set_ylabel('y [µm]', fontsize=10)
                panel.figure.patch.set_facecolor('#ffffff')
                panel.canvas.draw_idle()
            panel.apply_theme(self.theme)

        except Exception as exc:
            QMessageBox.warning(
                self,
                '3D renderer unavailable',
                'The OpenGL/VTK renderer could not start: '
                + str(exc)
            )


    def _sweep_tab(self):
        w = QWidget(); l = QVBoxLayout(w); l.setContentsMargins(10, 10, 10, 10)
        bar = QHBoxLayout(); bar.setSpacing(8)
        self.sw_param = QComboBox()
        self.sw_metric = QComboBox()
        # Visible labels are capitalized; internal metric codes stay unchanged
        metrics = [
            ('Peak intensity [W/cm²]', 'peak intensity [W/cm²]'),
            ('Peak position z [µm]', 'Peak position z [µm]'),
            ('FWHM X [nm]', 'FWHM x [nm]'),
            ('FWHM Y [nm]', 'FWHM y [nm]'),
            ('FWHM Z [nm]', 'FWHM z [nm]'),
            ('Longitudinal field share |Ez|²', '|Ez|² share')
        ]
        self.sw_metric.clear()
        for display, internal in metrics:
            self.sw_metric.addItem(display, internal)
        self.sw_from = Spin(0.0, -1e5, 1e5, 4); self.sw_to = Spin(1.0, -1e5, 1e5, 4); self.sw_n = ISpin(21, 2, 501)
        for wdg in (self.sw_from, self.sw_to): wdg.setFixedWidth(90)
        self.sw_quality = QComboBox(); self.sw_quality.addItems(['draft (fast)', 'full quality'])
        self.sw_go = QPushButton('Run sweep'); self.sw_go.setObjectName('primary'); self.sw_go.clicked.connect(self.run_sweep)
        self.sw_stop = QPushButton('Stop'); self.sw_stop.clicked.connect(lambda: self.sweep_worker and self.sweep_worker.stop())
        # Two readable rows instead of a single clipped toolbar.
        for label, widget in (
            ('Vary', self.sw_param), ('From', self.sw_from),
            ('To', self.sw_to), ('Steps', self.sw_n),
        ):
            bar.addWidget(QLabel(label))
            bar.addWidget(widget)
        bar.addStretch(1)
        l.addWidget(self._horizontal_strip(bar))
        actions = QHBoxLayout()
        actions.setSpacing(9)
        actions.addWidget(QLabel('Plot'))
        actions.addWidget(self.sw_metric)
        actions.addWidget(self.sw_quality)
        actions.addWidget(self.sw_go)
        actions.addWidget(self.sw_stop)
        actions.addStretch(1)
        l.addWidget(self._horizontal_strip(actions))
        self.sw_plot = LinePlot('', ''); l.addWidget(self.sw_plot, 1)
        self.sw_param.currentTextChanged.connect(self._sweep_defaults)
        return w

    # ------------------------------------------------------------------ theme
    def apply_theme(self, name):
        self.theme = name; self.settings.setValue('theme', name)
        QApplication.instance().setStyleSheet(stylesheet(name))
        if hasattr(self, 'theme_btn'):
            self.theme_btn.setIcon(_theme_toggle_icon(name))
        for m in self.findChildren(MapView): m.apply_theme(name)
        for m in self.findChildren(LinePlot): m.apply_theme(name)
        for d in self.findChildren(StackDiagram): d.theme = name; d.update()
        p = PALETTES[name]
        self.budget_plot.setBackground(p['plot_bg'])
        self._update_response_legends()
        self.render()

    # ------------------------------------------------------------------ scene <-> UI

    def set_scene(self, s: Scene):
        self._loading = True
        try:
            self.scene = copy.deepcopy(s)
            names = [o['name'] for o in presets.OBJECTIVES]
            self.obj_combo.setCurrentText(s.objective if s.objective in names else 'Custom')
            self.focal.setValue(s.focal_length); self.wl.setValue(s.wavelength)
            # Restore polarization, including custom states
            pol_code = {'custom': 'custom (elliptical)', 'custom-cylindrical': 'custom (cylindrical)'}.get(s.polarization, s.polarization)
            pol_index = self.pol.findData(pol_code)
            if pol_index < 0:
                raise ValueError(f'Unknown polarization in scene: {s.polarization}')
            self.pol.setCurrentIndex(pol_index)

            self.gamma.setValue(s.gamma); self.beta.setValue(s.beta); self.delta.setValue(s.delta)
            self.vortex.setValue(s.vortex)
            self.uniform.setChecked(s.fill <= 0); self.fill.setValue(s.fill if s.fill > 0 else 1.0)
            self.power.setValue(s.power_mW)
            self.stack.wavelength = s.wavelength; self.stack.set_layers(s.layers); self.zout.setValue(s.z_out)
            self.na.setRange(0.01, max(0.02, s.layers[0].n_re * 0.999))
            self.na.setValue(s.na)
            self.lateral.setValue(s.lateral); self.axial.setValue(s.axial); self.x0.setValue(s.x0); self.y0.setValue(s.y0)
            self.nxy.setValue(s.nxy); self.nz.setValue(s.nz)
            self.quality.setCurrentText(s.quality); self.ntheta.setValue(s.ntheta); self.nphi.setValue(s.nphi)
            self.do_volume.setChecked(s.volume); self.vxy.setValue(s.volume_xy); self.vz.setValue(s.volume_z)
            self._update_z0_range(); self.z0.setValue(s.z0)
            self.pupil_label.setText('Pupil: generated' if not s.pupil_files else 'Pupil: imported ' + Path(s.pupil_files[0]).name)

        finally:
            self._loading = False
        self.changed()

    def _update_z0_range(self):
        # Limit the XY plane position to half the axial size
        half_axial = self.axial.value() / 2.0
        self.z0.setRange(-half_axial, half_axial)

    def read_scene(self):
        s = self.scene
        s.na = self.na.value(); s.focal_length = self.focal.value(); s.wavelength = self.wl.value()
        p = self.pol.currentData()
        s.polarization = {'custom (elliptical)': 'custom', 'custom (cylindrical)': 'custom-cylindrical'}.get(p, p)
        s.gamma, s.beta, s.delta = self.gamma.value(), self.beta.value(), self.delta.value()
        s.vortex = self.vortex.value(); s.fill = 0.0 if self.uniform.isChecked() else self.fill.value()
        s.power_mW = self.power.value()
        s.layers = copy.deepcopy(self.stack.layers); s.z_out = self.zout.value()
        s.z0, s.x0, s.y0 = self.z0.value(), self.x0.value(), self.y0.value()
        s.lateral, s.axial = self.lateral.value(), self.axial.value()
        s.nxy, s.nz = self.nxy.value(), self.nz.value()
        s.quality = self.quality.currentText(); s.ntheta, s.nphi = self.ntheta.value(), self.nphi.value()
        s.volume = self.do_volume.isChecked(); s.volume_xy, s.volume_z = self.vxy.value(), self.vz.value()
        # Stack pre-compensation is disabled in this GUI
        s.correct = False
        s.correct_z = 0.0
        return s

    def changed(self, *_):
        if self._loading: return
        p = self.pol.currentData()
        self.custom_pol.setVisible(p == 'custom (elliptical)'); self.custom_cyl.setVisible(p == 'custom (cylindrical)')
        self.fill.setVisible(not self.uniform.isChecked())
        self.custom_q.setVisible(self.quality.currentText() == 'custom')
        if abs(self.stack.wavelength - self.wl.value()) > 1e-9:
            self.stack.refresh_dispersion(self.wl.value())
        self._update_z0_range()
        s = self.read_scene()
        n1 = s.layers[0].n_re
        if not math.isfinite(n1) or n1 <= 0:
            self.n_note.setText('Invalid immersion index: n must be positive')
            self.status_label.setText('⚠ Immersion refractive index must be positive.')
            return
        self.na.setRange(0.01, max(0.02, n1*0.999))
        self.n_note.setText(f'Immersion index n₁ = {n1:g} · θmax = '
                            f'{math.degrees(math.asin(min(s.na/n1, 1))):.1f}°')
        self.big_diagram.set_state(s.layers, s.z_out, s.z0, s.na, n1)
        nz = sum(1 for v in s.zernike if v)
        self.zern_label.setText('Aberrations: none' if nz == 0 else f'Aberrations: {nz} Zernike term' + ('s' if nz > 1 else ''))
        self._sweep_param_list()
        self._update_scene_modified()
        self._update_calculation_summary()
        self.schedule_preview()

    def schedule_preview(self):
        if self.worker is not None and self.worker.isRunning() and not self.worker.preview:
            return
        if self.live.isChecked():
            self.preview_timer.start()

    def objective_chosen(self, i):
        if i >= len(presets.OBJECTIVES):
            self.scene.objective = 'Custom'
            self.changed()
            return
        o = presets.OBJECTIVES[i]
        self._loading = True
        self.scene.objective = o['name']
        self.focal.setValue(o['f'])
        layers = self.stack.layers
        name = {1.0: 'Air', 1.334: 'Water', 1.518: 'Immersion Oil'}.get(o['n'], 'Custom')
        if name != 'Custom':
            nk = complex(presets.MATERIALS[name][0](self.wl.value()))
            layers[0] = Layer(name, nk.real, nk.imag, None)
        else:
            layers[0] = Layer('Custom', o['n'], 0.0, None)
        self.stack.set_layers(layers)
        self.na.setRange(0.01, max(0.02, layers[0].n_re * 0.999))
        self.na.setValue(o['na'])        
        self._loading = False
        self.changed()


    def objective_modified(self, *_):
        # Switch to Custom when a preset objective is manually modified
        if self._loading:
            return

        selected_name = self.obj_combo.currentText()

        objective = next(
            (
                obj for obj in presets.OBJECTIVES
                if obj['name'] == selected_name
            ),
            None
        )

        if objective is None:
            return

        na_changed = not math.isclose(
            self.na.value(),
            objective['na'],
            rel_tol=0.0,
            abs_tol=1e-6
        )

        focal_changed = not math.isclose(
            self.focal.value(),
            objective['f'],
            rel_tol=0.0,
            abs_tol=1e-6
        )

        if na_changed or focal_changed:
            self.obj_combo.blockSignals(True)
            self.obj_combo.setCurrentText('Custom')
            self.obj_combo.blockSignals(False)

            self.scene.objective = 'Custom'


    def import_pupil(self):
        files = []
        for what in ('Ex (complex matrix, rows = y)', 'Ey (complex matrix, rows = y)', 'x coordinates [µm]', 'y coordinates [µm]'):
            f, _ = QFileDialog.getOpenFileName(self, f'Pupil field: {what}', '', 'Text (*.txt *.dat *.csv);;All files (*)')
            if not f: return
            files.append(f)

        self.scene.pupil_files = files
        # Automatically select high accuracy for imported pupil fields
        self.quality.blockSignals(True)
        self.quality.setCurrentText('high')
        self.quality.blockSignals(False)

        self.pupil_label.setText('Pupil: imported ' + Path(files[0]).name + ' …')
        self.changed()

    def clear_pupil(self):
        self.scene.pupil_files = None; self.pupil_label.setText('Pupil: generated'); self.changed()

    def edit_zernike(self):
        d = ZernikeDialog(self.scene.zernike, self)
        if d.exec():
            self.scene.zernike = d.values(); self.changed()

    # ------------------------------------------------------------------ running
    def start(self, preview):
        if not preview:
            self.preview_timer.stop()
            self.pending = None 
        try:
            s = self.read_scene();
            s.validate()

            # Volume safety check
            voxel_count = int(s.volume_xy)**2 * int(s.volume_z)

            if not preview and s.volume and voxel_count > 2_000_000:
                QMessageBox.warning(self, 'Volume too large', f'The requested volume contains {voxel_count:,} voxels.\nMaximum allowed: 2,000,000 voxels.\n\nReduce Volume samples XY or Z.')
                return


        except Exception as exc:
            self.status_label.setText(f'⚠ {exc}'); return
        if self.worker is not None and self.worker.isRunning():
            if preview:
                if not self.worker.preview: return
                self.pending = 'preview'; return
            self.worker.stop()
        w = Worker(s, preview, check=not preview)
        w.done.connect(self.finished); w.failed.connect(self.failed)
        w.progress.connect(lambda f: self.progress.setValue(int(100*f)))
        w.finished.connect(self._worker_exit)
        self.worker = w
        self.status_label.setText('Live preview…' if preview else 'Computing full-quality solution with accuracy control…')
        self.compute_btn.setEnabled(preview)
        w.start()

    def _worker_exit(self):
        self.compute_btn.setEnabled(True)
        if self.pending:
            self.pending = None; self.start(preview=True)

    def finished(self, r):
        self.run_result = r
        self._last_run_scene_json = r.scene.to_json()
        self._update_calculation_summary()
        self.render()

    def failed(self, msg):
        self.status_label.setText(f'⚠ {msg}')

    # ------------------------------------------------------------------ rendering
    def interfaces(self, r):
        st = r.config.stack
        if len(st.indices) == 1: return []
        names = [l.name.split(' (')[0] for l in r.scene.layers]
        out = []
        lo, hi = float(r.z.min()), float(r.z.max())
        for k, z in enumerate(st.interfaces):
            if lo <= z <= hi:
                out.append((float(z), f'{names[k]} | {names[k+1]}'))
        return out

    def render(self):
        r = self.run_result
        if r is None: return
        tab = self.tabs.currentIndex()
        try:
            if tab == 0:
                self.render_pupil(r)
            elif tab == 1:
                self.render_response(r)
            elif tab == 2:
                self.render_focus(r)
            elif tab == 3:
                self.render_components(r)
            elif tab == 4:
                self.render_polarization(r)
            elif tab == 5:
                self.render_energy(r)
            elif tab == 6:
                self.render_volume(r)
        except Exception as exc:
            traceback.print_exc(); self.status_label.setText(f'⚠ display: {exc}')
        self.render_tiles(r)
        self._schedule_plot_autorange()

    def _schedule_plot_autorange(self):
        """Run after the final Qt layout pass (including changed tab sizes)."""
        if not getattr(self, '_autorange_pending', False):
            self._autorange_pending = True
            QTimer.singleShot(0, self._refresh_current_plot_ranges)

    def _refresh_current_plot_ranges(self):
        self._autorange_pending = False
        if self.run_result is None:
            return
        tab = self.tabs.currentIndex()
        groups = {
            0: (self.pup_I, self.pup_Iy, self.pup_P, self.pup_Py),
            2: (self.map_xz, self.map_xy),
            3: tuple(self.comp_maps),
            4: (self.ell_map, *self.pol_maps),
            5: (self.e_sxz, self.e_abs),
            6: tuple(self.vol_maps),
        }
        for m in groups.get(tab, ()):
            m.reframe_data()
        if tab == 1:
            for curve in (self.resp_R, self.resp_amp):
                plot = curve.getPlotItem()
                plot.autoRange(padding=0.03)
                vb = plot.getViewBox()
                vb.forget_home()
                vb.remember_home()
        elif tab == 2:
            r = self.run_result
            # Horizontal axes always cover the full calculated profiles;
            # their Y range is set independently to map extrema + 15%.
            if self.focus_axial_plane.currentIndex() == 1:
                transverse = r.y
            else:
                transverse = r.x
            lat_range = (min(float(r.x[0]), float(r.y[0])),
                         max(float(r.x[-1]), float(r.y[-1])))
            for curve, x in ((self.prof_ax, r.z), (self.prof_lat, lat_range)):
                curve.setXRange(float(x[0]), float(x[-1]), padding=0)
                curve.getViewBox().forget_home()
                curve.getViewBox().remember_home()

    def render_tiles(self, r):
        m, b = r.metrics, r.budget
        p = PALETTES[self.theme]
        self.t_peak.set(f'{m["E_peak_Vm"]:.3g}')
        self.t_int.set(f'{m["I_peak_Wcm2"]:.3g}')
        self.t_zpk.set(f'{m["z_peak"]:+.3f}')
        def nm(v): return '—' if not np.isfinite(v) else f'{1000*v:.0f}'
        self.t_fw.set(f'{nm(m["fwhm_x"])} · {nm(m["fwhm_y"])} · {nm(m["fwhm_z"])}')
        self.t_ez.set(f'{100*m["frac"][2]:.1f} %')
        A = sum(b['A'])
        self.t_rta.set(f'{100*b["R"]:.1f} · {100*b["T"]:.1f} · {100*A:.1f} %')
        if r.preview or r.sampling_error is None:
            self.t_acc.set('preview', p['muted'])
        else:
            e = r.sampling_error
            self.t_acc.set(f'{e:.0e}', p['ok'] if e < 1e-3 else p['warn'] if e < 1e-2 else p['bad'])

    def _display(self, r, plane, name):
        vals, unit = quantity(r, plane, name)
        if name.startswith('arg'):
            return vals, 'phase', (-np.pi, np.pi), 'phase [rad]', False
        absolute = self.norm.currentText() == 'absolute units'
        return vals, None, None, unit if absolute else 'normalized', absolute


    def render_focus(self, r):
        name = self.qty.currentText()
        cmap = self.cmap.currentText()
        log = self.logscale.isChecked()
        use_yz = self.focus_axial_plane.currentIndex() == 1

        # The YZ magnetic field is not stored in the result.
        if use_yz and name == 'Sz (Poynting)':
            self.focus_axial_plane.blockSignals(True)
            self.focus_axial_plane.setCurrentIndex(0)
            self.focus_axial_plane.blockSignals(False)
            use_yz = False
            self.status_label.setText(
                'Sz in YZ is unavailable (Hyz not computed); showing XZ.'
            )

        # Reuse the already computed axial electric-field sections
        if use_yz:
            r_ax = SimpleNamespace(
                Exz=r.Eyz,
                Hxz=None,
                scale=r.scale,
                config=r.config,
                scene=r.scene,
                z=r.z
            )
            transverse = r.y
            axis_name = 'Y'
        else:
            r_ax = r
            transverse = r.x
            axis_name = 'X'

        axial, _, _, label, absolute = self._display(
            r_ax, 'xz', name
        )
        xy, *_ = self._display(r, 'xy', name)

        signed = name == 'Sz (Poynting)'

        # Preserve the existing normalization options
        if not absolute:
            if self.shared.isChecked():
                common = max(
                    float(np.max(np.abs(axial))),
                    float(np.max(np.abs(xy))),
                    1e-300
                )
            else:
                common = None

            axial = axial / (
                common or max(float(np.max(np.abs(axial))), 1e-300)
            )
            xy = xy / (
                common or max(float(np.max(np.abs(xy))), 1e-300)
            )

        if signed:
            peak = max(
                float(np.max(np.abs(axial))),
                float(np.max(np.abs(xy))),
                1e-30
            )
            levels_ax = levels_xy = (-peak, peak)
            cm = 'diverging'
            log = False
        else:
            cm = cmap

            if self.shared.isChecked():
                peak = max(
                    float(np.max(axial)),
                    float(np.max(xy)),
                    1e-30
                )
                levels_ax = levels_xy = (0, peak)
            else:
                levels_ax = (
                    0, max(float(np.max(axial)), 1e-30)
                )
                levels_xy = (
                    0, max(float(np.max(xy)), 1e-30)
                )

        # Axial map
        # Allow independent X and Z scaling
        self.map_xz.plot.setAspectLocked(False)
        self.map_xz.plot.setLabel(
            'bottom', f'{axis_name.lower()} [µm]'
        )

        self.map_xz.set_data(
            axial, transverse, r.z, cm,
            None if log else levels_ax, log, label
        )
        self.map_xz.set_title(
            f'{name} · {axis_name}Z section'
        )
        self.map_xz.set_interfaces(self.interfaces(r))


        # The plot's graphics layout follows the actual tab geometry.
        # Never freeze a column/row to the size from an earlier render.

        # XY map
        self.map_xy.set_data(
            xy, r.x, r.y, cm,
            None if log else levels_xy, log, label
        )
        self.map_xy.set_title(
            f'{name} · XY plane (z = {r.scene.z0:g} [µm])'
        )

        # Configure sliders using the already computed samples
        for slider, count in (
            (self.focus_xy_x_slider, len(r.x)),
            (self.focus_xy_y_slider, len(r.y)),
            (self.focus_axial_slider, len(transverse)),
            (self.focus_axial_z_slider, len(r.z))
        ):
            old_max = slider.maximum()
            old_value = slider.value()

            slider.blockSignals(True)
            slider.setRange(0, count - 1)

            if old_max == 0:
                value = count // 2
            else:
                value = round(
                    old_value * (count - 1) / old_max
                )

            slider.setValue(value)
            slider.blockSignals(False)

        # Cache displayed quantities for live profile extraction
        self._focus_transverse = transverse
        self._focus_xy_values = xy
        self._focus_axial_values = axial

        self.prof_lat.setLabel('left', label)
        self.prof_ax.setLabel('left', label)
        self.prof_ax.setLabel('bottom', 'position [µm]')        

        if log:
            def log_values(values):
                positive = values[values > 0]
                floor = (
                    float(positive.max()) * 1e-4
                    if positive.size else 1e-30
                )
                return np.log10(np.maximum(values, floor))

            self._focus_xy_values = log_values(xy)
            self._focus_axial_values = log_values(axial)

            self.prof_lat.setLabel('left', f'log10 {label}')
            self.prof_ax.setLabel('left', f'log10 {label}')

        self._focus_update_profiles()


    def _focus_update_profiles(self, *_):
        r = self.run_result

        if r is None or not hasattr(self, '_focus_xy_values'):
            return

        ix = min(self.focus_xy_x_slider.value(), len(r.x) - 1)
        iy = min(self.focus_xy_y_slider.value(), len(r.y) - 1)
        iz = min(self.focus_axial_z_slider.value(), len(r.z) - 1)        

        transverse = self._focus_transverse
        it = min(
            self.focus_axial_slider.value(),
            len(transverse) - 1
        )

        # Update numerical position labels
        self.focus_xy_x_label.setText(
            f'X = {r.x[ix]:+.3f} [µm]'
        )
        self.focus_xy_y_label.setText(
            f'Y = {r.y[iy]:+.3f} [µm]'
        )

        axis_name = (
            'X' if self.focus_axial_plane.currentIndex() == 0
            else 'Y'
        )

        self.focus_axial_label.setText(
            f'{axis_name} = {transverse[it]:+.3f} [µm]'
        )
        self.focus_axial_z_label.setText(f'Z = {r.z[iz]:+.3f} [µm]')
        # Move the extraction lines on the corresponding maps
        self.focus_xy_vline.setValue(float(r.x[ix]))
        self.focus_xy_hline.setValue(float(r.y[iy]))
        self.focus_axial_vline.setValue(float(transverse[it]))
        self.focus_axial_hline.setValue(float(r.z[iz]))

        colors = PALETTES[self.theme]
        xy = self._focus_xy_values
        axial = self._focus_axial_values

        # Orthogonal X and Y profiles
        self.prof_lat.clear()
        self.prof_lat.legend.clear()

        self.prof_lat.plot(
            r.x, xy[iy, :],
            pen=pg.mkPen(colors['accent'], width=2)
        )

        self.prof_lat.plot(
            r.y, xy[:, ix],
            pen=pg.mkPen(colors['accent2'], width=2)
        )

        # Axial Z profile at the chosen transverse position
        self.prof_ax.clear()
        self.prof_ax.legend.clear()

        self.prof_ax.plot(transverse, axial[iz, :], pen=pg.mkPen(colors['accent'], width=2))
        self.prof_ax.plot(r.z, axial[:, it], pen=pg.mkPen(colors['accent2'], width=2))

        # Constant profile Y ranges are calculated from the complete map,
        # never from the current slider position. Signed and logarithmic
        # quantities retain their physical lower bound.
        for profile, full_map in (
            (self.prof_ax, axial), (self.prof_lat, xy)
        ):
            finite = np.asarray(full_map)[np.isfinite(full_map)]
            if finite.size:
                low = float(finite.min())
                high = float(finite.max())
                if high <= low:
                    delta = max(abs(high) * 0.01, 1e-12)
                    low -= delta
                    high += delta
                # A fixed map-wide scale preserves relative profile heights
                # while allowing a little room above the global maximum.
                high += 0.15 * (high - low)
                profile.getPlotItem().enableAutoRange(axis='y', enable=False)
                profile.getPlotItem().setYRange(low, high, padding=0)
                profile.getViewBox().forget_home()
                profile.getViewBox().remember_home()

        # Mark stack interfaces on the axial profile
        for z, _ in self.interfaces(r):
            self.prof_ax.addItem(
                pg.InfiniteLine(
                    pos=z,
                    angle=90,
                    pen=pg.mkPen(
                        colors['muted'],
                        style=Qt.PenStyle.DashLine
                    )
                )
            )

        # Mark the position of the XY observation plane
        self.prof_ax.addItem(
            pg.InfiniteLine(
                pos=r.scene.z0,
                angle=90,
                pen=pg.mkPen(colors['accent'], width=1)
            )
        )


    def render_components(self, r):
        # Components are displayed in the XY plane only
        xy = True
        E = r.Exy
        ycoord = r.y

        P = np.abs(E)**2; hi = P.sum(0).max()
        for i in range(3):
            m = self.comp_maps[i]; m.plot.setLabel('left', 'y [µm]' if xy else 'z [µm]')
            m.set_data(P[i]/hi, r.x, ycoord, self.cmap.currentText(),
                       (0, 1) if self.comp_shared.isChecked() else (0, max(P[i].max()/hi, 1e-30)), False, 'Normalized intensity')
            frac = P[i].sum()/P.sum()
            m.set_title(f'|E{"xyz"[i]}|²  ({100*frac:.1f} %)')
            m.set_interfaces([] if xy else self.interfaces(r))
        for i in range(3):
            m = self.comp_maps[3+i]; m.plot.setLabel('left', 'y [µm]' if xy else 'z [µm]')
            ph = np.angle(E[i]); ph = np.where(P[i] > 1e-6*hi, ph, np.nan)
            m.set_data(np.nan_to_num(ph), r.x, ycoord, 'phase', (-np.pi, np.pi), False, 'Phase [rad]')
            m.set_title(f'arg E{"xyz"[i]}')
            m.set_interfaces([] if xy else self.interfaces(r))


    def render_polarization(self, r):
        E = r.Exy

        intensity = np.sum(np.abs(E)**2, axis=0)
        peak = max(float(np.nanmax(intensity)), 1e-30)

        threshold = self.pol_ellipse_threshold.value() / 100.0

        # One shared intensity mask for every polarization map.
        pol = polarization_ellipse_3d(E, threshold=threshold)

        mask = (
            pol.valid
            & (intensity > 0)
            & np.isfinite(intensity)
        )

        self._pol_result = r
        self._pol_ellipse = pol

        # Map 1: intensity background and polarization ellipses.
        background = np.where(
            mask,
            intensity / peak,
            np.nan
        )

        self.ell_map.set_data(
            background,
            r.x, r.y,
            self.cmap.currentText(),
            (0, 1),
            False,
            'Normalized intensity'
        )

        self.ell_map.bar.axis.setTicks([[
            (0, '0'),
            (0.25, '0.25'),
            (0.5, '0.5'),
            (0.75, '0.75'),
            (1, '1')
        ]])

        self.ell_map.set_title(
            '3D polarization ellipses (XY projection)'
        )

        # Map 2: ellipticity.
        major = np.linalg.norm(pol.major, axis=0)
        minor = np.linalg.norm(pol.minor, axis=0)

        ellipticity = np.divide(
            minor,
            major,
            out=np.zeros_like(major),
            where=major > 0
        )

        ellipticity = np.where(
            mask,
            ellipticity,
            np.nan
        )

        # Map 3: normalized spin density.
        spin = np.imag(
            np.conj(E[0]) * E[1]
            - np.conj(E[1]) * E[0]
        ) / np.maximum(intensity, 1e-300)

        spin = np.where(
            mask,
            spin,
            np.nan
        )

        maps = [
            (
                self.pol_maps[0],
                ellipticity,
                'Ellipticity |B|/|A|',
                'viridis',
                (0, 1),
                'Ellipticity'
            ),
            (
                self.pol_maps[1],
                spin,
                'Spin density Im(E*×E)z / |E|²',
                'diverging',
                (-1, 1),
                'Spin density'
            ),
        ]

        for m, data, title, cmap, levels, label in maps:
            m.set_data(
                data,
                r.x, r.y,
                cmap,
                levels,
                False,
                label
            )

            if levels[0] < 0:
                ticks = [
                    (-1, '-1'),
                    (-0.5, '-0.5'),
                    (0, '0'),
                    (0.5, '0.5'),
                    (1, '1')
                ]
            else:
                ticks = [
                    (0, '0'),
                    (0.25, '0.25'),
                    (0.5, '0.5'),
                    (0.75, '0.75'),
                    (1, '1')
                ]

            m.bar.axis.setTicks([ticks])
            m.set_title(title)

        # Update ellipse overlays using the same threshold.
        self._redraw_pol_ellipses()

        # Update the A/B/N dialog if it is open.
        if (
            self.pol_abn_dialog is not None
            and self.pol_abn_dialog.isVisible()
        ):
            self._draw_abn_maps(r, pol)


    def _redraw_pol_ellipses(self, *_):
        # Update labels even when no result is available
        density = self.pol_ellipse_density.value()
        size_fraction = self.pol_ellipse_size.value() / 100.0

        self.pol_density_label.setText(f'{density} per axis')
        self.pol_size_label.setText(
            f'{self.pol_ellipse_size.value()}%'
        )

        r = self._pol_result
        pol = self._pol_ellipse

        if r is None or pol is None:
            return

        # Remove previous ellipses
        for item in self.ellipse_items:
            self.ell_map.plot.removeItem(item)

        self.ellipse_items = []

        nx = len(r.x)
        ny = len(r.y)

        # Sample positions across both image axes
        count = min(density, nx - 2, ny - 2)

        if count < 1:
            return

        ix_list = np.unique(
            np.rint(
                np.linspace(0, nx - 1, count + 2)
            ).astype(int)[1:-1]
        )

        iy_list = np.unique(
            np.rint(
                np.linspace(0, ny - 1, count + 2)
            ).astype(int)[1:-1]
        )

        # Ellipse radius relative to sampling spacing
        spacing = min(
            float(np.ptp(r.x)),
            float(np.ptp(r.y))
        ) / (count + 1)

        ellipse_size = spacing * size_fraction

        t = np.linspace(0, 2*np.pi, 64)
        ct = np.cos(t)
        st = np.sin(t)

        # Same color for all ellipses
        pen = pg.mkPen('#74E3FF', width=1.5)

        for iy in iy_list:
            for ix in ix_list:

                if not pol.valid[iy, ix]:
                    continue

                A = pol.major[:, iy, ix]
                B = pol.minor[:, iy, ix]

                major_length = np.linalg.norm(A)

                if major_length <= 0:
                    continue

                # Project the local 3D ellipse onto XY
                curve = (
                    A[:, None] * ct[None, :]
                    + B[:, None] * st[None, :]
                ) / major_length * ellipse_size

                item = pg.PlotCurveItem(
                    r.x[ix] + curve[0],
                    r.y[iy] + curve[1],
                    pen=pen
                )

                self.ell_map.plot.addItem(
                    item,
                    ignoreBounds=True
                )
                self.ellipse_items.append(item)

    def _show_abn_dialog(self):
        r = self.run_result

        if r is None:
            QMessageBox.information(
                self,
                'Polarization components',
                'Compute the focal field first.'
            )
            return

        # Create the window only once
        if self.pol_abn_dialog is None:
            dialog = QDialog(self)
            dialog.setWindowTitle(
                'Polarization ellipse components - A, B, N'
            )
            screen = self.screen() or QGuiApplication.primaryScreen()
            if screen is not None:
                available = screen.availableGeometry()
                dialog.resize(min(1024, int(available.width() * 0.90)),
                              min(720, int(available.height() * 0.90)))
            else:
                dialog.resize(1024, 720)

            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(12, 12, 12, 12)

            note = QLabel(
                'A: major axis | B: minor axis | '
                'N: ellipse normal (A × B). '
                'A and B use a common amplitude normalization.'
            )
            note.setWordWrap(True)
            layout.addWidget(note)

            grid = QGridLayout()
            grid.setSpacing(12)

            self.pol_abn_maps = []

            for row in range(3):
                map_row = []

                for col in range(3):
                    m = MapView('x [µm]', 'y [µm]')

                    grid.addWidget(m, row, col)
                    map_row.append(m)

                self.pol_abn_maps.append(map_row)

            for i in range(3):
                grid.setRowStretch(i, 1)
                grid.setColumnStretch(i, 1)

            layout.addLayout(grid, 1)

            self.pol_abn_dialog = dialog

        self.pol_abn_dialog.show()
        self.pol_abn_dialog.raise_()
        self.pol_abn_dialog.activateWindow()

        self._draw_abn_maps(r, self._pol_ellipse or polarization_ellipse_3d(
            r.Exy, threshold=self.pol_ellipse_threshold.value() / 100.0
        ))


    def _draw_abn_maps(self, r, pol):
        # Match the normalization of the previous application
        ab_max = max(
            float(np.max(np.abs(pol.major))),
            float(np.max(np.abs(pol.minor))),
            np.finfo(float).eps
        )

        n_max = max(
            float(np.max(np.abs(pol.normal))),
            np.finfo(float).eps
        )

        vector_groups = [
            pol.major / ab_max,
            pol.minor / ab_max,
            pol.normal
        ]

        names = ['A', 'B', 'N']
        components = ['x', 'y', 'z']

        for row in range(3):
            vectors = vector_groups[row]

            # A and B are normalized to a common maximum.
            # N is displayed in its original amplitude.
            limit = n_max if row == 2 else 1.0

            label = (
                'Amplitude'
                if row == 2
                else 'Normalized amplitude'
            )

            for col in range(3):
                m = self.pol_abn_maps[row][col]

                # Hide components outside the valid polarization region.
                visible = (
                    pol.valid
                    & (np.sum(np.abs(r.Exy)**2, axis=0) > 0)
                )

                m.set_data(
                    np.where(visible, vectors[col], np.nan),
                    r.x,
                    r.y,
                    'diverging',
                    (-limit, limit),
                    False,
                    label
                )

                m.set_title(
                    f'{names[row]}{components[col]}'
                )


    def render_energy(self, r):
        sxz, u = quantity(r, 'xz', 'Sz (Poynting)')
        a = np.abs(sxz).max() or 1
        self.e_sxz.set_data(sxz, r.x, r.z, 'diverging', (-a, a), False, 'Energy flux [W/cm²]'); self.e_sxz.set_title('Poynting Sz in the XZ section')
        self.e_sxz.bar.axis.tickStrings = lambda values, scale, spacing: [f'{v:.1e}' for v in values]
        self.e_sxz.bar.axis.setWidth(65)
        self.e_sxz.bar.axis.update()
        pa, u = quantity(r, 'xz', 'Absorbed power')
        self.e_abs.set_data(pa, r.x, r.z, 'inferno', (0, max(pa.max(), 1e-30)), False, 'Absorbed power density [W/cm³]')
        self.e_abs.set_title('Absorption — XZ' + (' — no absorber' if pa.max() <= 0 else ''))
        interface_color = 'k' if self.theme == 'light' else 'w'
        self.e_sxz.set_interfaces(self.interfaces(r), line_color=interface_color, label_color=interface_color)

        # Keep the three original map panels fully flexible on resize.

        b = r.budget; names = [l.name.split(' (')[0] for l in r.scene.layers]
        parts = [('reflected', b['R'], '#6b9bff')] + [(f'absorbed in {names[j]}', a_, '#ff8a4c')
                                                     for j, a_ in enumerate(b['A']) if a_ > 1e-6] + [('transmitted', b['T'], '#3ccf7e')]
        self.budget_plot.clear()
        x0 = 0
        p = PALETTES[self.theme]
        for k, (lab, v, col) in enumerate(parts):
            bar = pg.BarGraphItem(x0=[x0], y=[0], width=[v], height=0.6, brush=col, pen=pg.mkPen(p['plot_bg']))
            self.budget_plot.addItem(bar)
            if v > 0.04:
                t = pg.TextItem(f'{lab}\n{100*v:.1f} %', color=p['plot_fg'], anchor=(0.5, 0.5)); t.setPos(x0+v/2, 0)
                self.budget_plot.addItem(t)
            x0 += v
        self.budget_plot.setXRange(0, 1, padding=0.01); self.budget_plot.setYRange(-0.5, 0.5)
        self.budget_plot.hideAxis('left'); self.budget_plot.setTitle('Power budget of the focused beam', color=p['plot_fg'])
        self.budget_plot.getViewBox().forget_home()



    def _update_response_legends(self):
        if not hasattr(self, 'response_legend_labels'):
            return
        palette = PALETTES[self.theme]
        for widget, name in self.response_legend_labels:
            color = (palette['accent'] if name.startswith(('R ', '|r')) else
                     '#2bb673' if name.startswith(('T ', '|t')) else
                     palette['accent2'])
            marker = '┄┄' if name.endswith(' p') or name.endswith('p|') else '━━'
            widget.setText(f'<span style="color:{color}">{marker}</span> {name}')

    def render_response(self, r):
        st = r.config.stack
        th = np.linspace(0, math.asin(0.9999*min(1, r.scene.na/st.indices[0].real)), 600)
        pc = power_coefficients(st, th, r.scene.wavelength)
        deg = np.degrees(th); p = PALETTES[self.theme]
        self.resp_R.clear(); self.resp_R.legend.clear()
        for key, col, style, lab in (('Rs', p['accent'], Qt.PenStyle.SolidLine, 'R s'), ('Rp', p['accent'], Qt.PenStyle.DashLine, 'R p'),
                                     ('Ts', '#2bb673', Qt.PenStyle.SolidLine, 'T s'), ('Tp', '#2bb673', Qt.PenStyle.DashLine, 'T p'),
                                     ('As', p['accent2'], Qt.PenStyle.SolidLine, 'A s'), ('Ap', p['accent2'], Qt.PenStyle.DashLine, 'A p')):
            self.resp_R.plot(deg, pc[key], pen=pg.mkPen(col, width=2, style=style), name=lab)
        a = pc['response']
        self.resp_amp.clear(); self.resp_amp.legend.clear()
        for arr, col, style, lab in ((a.rs, p['accent'], Qt.PenStyle.SolidLine, '|r s|'), (a.rp, p['accent'], Qt.PenStyle.DashLine, '|r p|'),
                                     (a.ts, '#2bb673', Qt.PenStyle.SolidLine, '|t s|'), (a.tp, '#2bb673', Qt.PenStyle.DashLine, '|t p|')):
            self.resp_amp.plot(deg, np.abs(arr), pen=pg.mkPen(col, width=2, style=style), name=lab)
        n1 = st.indices[0].real
        for nj in st.indices[1:]:
            if abs(np.imag(nj)) < 1e-12 and np.real(nj) < n1 and np.real(nj)/n1 < r.scene.na/n1:
                tc = math.degrees(math.asin(np.real(nj)/n1))
                for plot in (self.resp_R, self.resp_amp):
                    plot.addItem(pg.InfiniteLine(pos=tc, angle=90, pen=pg.mkPen(p['muted'], style=Qt.PenStyle.DotLine),
                                                 label=f'θc {tc:.1f}°', labelOpts=dict(color=p['muted'], position=0.9)))



    def render_pupil(self, r):
        # Dense angular sampling for pupil display only
        display_cfg = r.scene.config(
            ntheta=r.config.sampling.theta,
            nphi=721,
            axial=False,
            volume=False
        )

        pupil = pupil_for(r.scene, display_cfg)

        if pupil is None:
            breakpoints = critical_angles(
                display_cfg.stack,
                float(np.real(display_cfg.stack.indices[0])),
                display_cfg.objective.theta_max
            )

            pupil = generate_inc_field(
                display_cfg.objective,
                display_cfg.illumination,
                display_cfg.sampling.phi,
                display_cfg.sampling.theta,
                quadrature='gauss',
                breakpoints=breakpoints
            )

        ex, ey, xp, yp = pupil

        # Preserve the closed azimuthal interval [-pi, pi]
        n_phi = ex.shape[0]
        mid = (n_phi - 1) // 2
        order = np.r_[mid:n_phi, 1:mid + 1]

        ex = ex[order, :]
        ey = ey[order, :]
        xp = xp[order, :]
        yp = yp[order, :]

        # Original polar coordinates
        phi = np.unwrap(np.arctan2(yp[:, -1], xp[:, -1]))
        rho = np.hypot(xp[0, :], yp[0, :]) / 1000.0

        radius = float(rho[-1])

        # Cartesian display grid, in millimeters
        grid = np.linspace(-radius, radius, 601)
        X, Y = np.meshgrid(grid, grid)

        R = np.hypot(X, Y)
        inside = R <= radius

        # Periodic angular coordinates
        Phi = (
            (np.arctan2(Y, X) - phi[0]) % (2.0 * np.pi)
            + phi[0]
        )

        # Extend the first and last radial samples for display
        R_sample = np.clip(R, rho[0], rho[-1])

        points = np.column_stack((
            Phi.ravel(),
            R_sample.ravel()
        ))

        # Interpolate complex fields in polar coordinates
        # before calculating intensity or phase
        interp_x = RegularGridInterpolator(
            (phi, rho),
            ex,
            method='linear',
            bounds_error=False,
            fill_value=None
        )

        interp_y = RegularGridInterpolator(
            (phi, rho),
            ey,
            method='linear',
            bounds_error=False,
            fill_value=None
        )

        Ex = interp_x(points).reshape(X.shape)
        Ey = interp_y(points).reshape(X.shape)

        # Component intensities with a common normalization
        Ix = np.abs(Ex)**2
        Iy = np.abs(Ey)**2

        peak = max(float(Ix.max()), float(Iy.max()), 1e-30)

        Ix = np.where(inside, Ix / peak, np.nan)
        Iy = np.where(inside, Iy / peak, np.nan)

        # Component phases
        # Hide undefined phase where intensity vanishes
        threshold = peak * 1e-10

        Px = np.where(
            inside & (np.abs(Ex)**2 > threshold),
            np.angle(Ex),
            np.nan
        )

        Py = np.where(
            inside & (np.abs(Ey)**2 > threshold),
            np.angle(Ey),
            np.nan
        )

        # PyQtGraph maps with native mouse interaction
        cmap = self.cmap.currentText()

        self.pup_I.set_data(
            Ix, grid, grid, cmap,
            (0, 1), False, 'Normalized intensity'
        )
        self.pup_I.set_title('|Ex|²')

        self.pup_Iy.set_data(
            Iy, grid, grid, cmap,
            (0, 1), False, 'Normalized intensity'
        )
        self.pup_Iy.set_title('|Ey|²')

        self.pup_P.set_data(
            Px, grid, grid, 'phase',
            (-np.pi, np.pi), False, 'Phase [rad]'
        )
        self.pup_P.set_title('arg(Ex)')

        self.pup_Py.set_data(
            Py, grid, grid, 'phase',
            (-np.pi, np.pi), False, 'Phase [rad]'
        )
        self.pup_Py.set_title('arg(Ey)')





    def render_volume(self, r):
        if r.volume is None:
            return
        V = np.sum(np.abs(r.volume)**2, 0)       # (z, y, x)
        pk = V.max() or 1
        # The volume controls contain only editable controls; instructions are in the tab description.
        self.vol_slider.setRange(float(r.vz[0]), float(r.vz[-1]))
        iz = int(np.argmin(abs(r.vz - self.vol_slider.value())))
        cm = self.cmap.currentText()
        normalize = self.vol_normalize.isChecked()
        slice_data = V[iz]/pk if normalize else V[iz]
        slice_max = 1 if normalize else (float(slice_data.max()) or 1e-30)
        self.vol_maps[0].set_data(slice_data, r.vx, r.vy, cm, (0, slice_max), False, 'normalized' if normalize else 'Intensity [a.u.]'); self.vol_maps[0].set_title(f'Slice z = {r.vz[iz]:.3f} [µm]')
        iz_best = int(np.argmax(V.max(axis=(1, 2))))
        z_best = float(r.vz[iz_best])
        self.vol_maps[1].set_data(V[iz_best]/pk, r.vx, r.vy, cm, (0, 1), False, 'normalized'); self.vol_maps[1].set_title(f'Best focus at z = {z_best:.3f} [µm]')
        
        # Select one axial projection from the computed volume
        axial_map = self.vol_maps[2]

        if self.vol_axial_plane.currentIndex() == 0:
            # XZ: maximum intensity projection along Y
            axial_values = V.max(axis=1) / pk
            transverse = r.vx
            axis_label = 'x [µm]'
            title = 'Maximum projection along Y (XZ)'
        else:
            # YZ: maximum intensity projection along X
            axial_values = V.max(axis=2) / pk
            transverse = r.vy
            axis_label = 'y [µm]'
            title = 'Maximum projection along X (YZ)'

        axial_map.plot.setLabel('bottom', axis_label)

        axial_map.set_data(
            axial_values,
            transverse,
            r.vz,
            cm,
            (0, 1),
            False,
            'normalized'
        )

        axial_map.set_title(title)


        fake = SimpleNamespace(z=r.vz, scene=r.scene, config=r.config)
        for m in self.vol_maps[2:]:
            m.set_interfaces(self.interfaces(fake)); m.set_marker(float(r.vz[iz]))

    def xz_clicked(self, x, z):
        self.z0.setValue(z); self.changed()

    # ------------------------------------------------------------------ sweeps
    def _sweep_param_list(self):
        cur = self.sw_param.currentData()
        items = ['XY plane z₀ [µm]', 'NA', 'Wavelength [µm]', 'Last interface z_out [µm]', 'Gaussian fill w₀/ρmax',
                 'Vortex charge']
        for i, l in enumerate(self.stack.layers[1:-1], start=1):
            items.append(f'Thickness of layer {i}: {l.name.split(" (")[0]} [µm]')
        existing = [
            self.sw_param.itemData(i)
            for i in range(self.sw_param.count())
        ]

        if items != existing:
            self.sw_param.blockSignals(True)
            self.sw_param.clear()

            for name in items:
                display_name = name[0].upper() + name[1:]
                self.sw_param.addItem(display_name, name)

            if cur in items:
                self.sw_param.setCurrentIndex(items.index(cur))

            self.sw_param.blockSignals(False)

    def _sweep_defaults(self, name):
        name = self.sw_param.currentData() or name
        s = self.scene
        ranges = {'NA': (0.3, 0.99*s.layers[0].n_re), 'Wavelength [µm]': (0.4, 0.8), 'Gaussian fill w₀/ρmax': (0.3, 3),
                  'Vortex charge': (-3, 3), 'XY plane z₀ [µm]': (s.z0-1, s.z0+1), 'Last interface z_out [µm]': (s.z_out-2, s.z_out+2)
                  }
        lo, hi = ranges.get(name, (0.0, 0.2))
        self.sw_from.setValue(lo); self.sw_to.setValue(hi)

    def _setter(self, name):
        if name.startswith('Thickness of layer'):
            j = int(name.split()[3].rstrip(':'))
            def f(s, v):
                old = s.layers[j].d; s.layers[j].d = max(0.0, v)
                if j == len(s.layers)-2: s.z_out += (s.layers[j].d - old)  # keep the bottom of the stack fixed
            return f
        
        def set_wavelength(s, v):
            s.wavelength = float(v)
            for layer in s.layers:
                entry = presets.MATERIALS.get(layer.name)
                if entry:
                    n = complex(entry[0](s.wavelength))
                    layer.n_re = round(n.real, 6)
                    layer.n_im = round(n.imag, 6)

        return {'XY plane z₀ [µm]': lambda s, v: setattr(s, 'z0', v), 'NA': lambda s, v: setattr(s, 'na', v),
                'Wavelength [µm]': set_wavelength,
                'Last interface z_out [µm]': lambda s, v: setattr(s, 'z_out', v),
                'Gaussian fill w₀/ρmax': lambda s, v: setattr(s, 'fill', v),
                'Vortex charge': lambda s, v: setattr(s, 'vortex', int(round(v))),
                }[name]

    def run_sweep(self):
        if self.sweep_worker is not None and self.sweep_worker.isRunning(): return
        s = copy.deepcopy(self.read_scene())
        self._sweep_scene = copy.deepcopy(s)
        s.volume = False
        name = self.sw_param.currentData()
        vals = np.linspace(self.sw_from.value(), self.sw_to.value(), self.sw_n.value())
        if name == 'Vortex charge': vals = np.unique(np.round(vals))
        self._sweep_export_details = {
            'varied_parameter': self.sw_param.currentText(),
            'output_metric': self.sw_metric.currentText(),
            'quality': self.sw_quality.currentText(),
            'requested_samples': int(len(vals)),
        }
        self.sw_x, self.sw_y = [], []
        metric = self.sw_metric.currentData()
        self.sw_plot.clear(); self.sw_plot.setLabel('bottom', self.sw_param.currentText()); self.sw_plot.setLabel('left', self.sw_metric.currentText())
        p = PALETTES[self.theme]
        self.sw_curve = self.sw_plot.plot([], [], pen=pg.mkPen(p['accent'], width=2), symbol='o', symbolSize=5,
                                          symbolBrush=p['accent'])
        def extract(d):
            m, b = d['metrics'], d['budget']
            j = next((i for i, l in enumerate(s.layers) if l.n_im > 0), None)
            return {'peak intensity [W/cm²]': m['I_peak_Wcm2'], 'Peak position z [µm]': m['z_peak'],
                    'FWHM x [nm]': 1000*m['fwhm_x'], 'FWHM y [nm]': 1000*m['fwhm_y'], 'FWHM z [nm]': 1000*m['fwhm_z'],
                    '|Ez|² share': m['frac'][2], 'R': b['R'], 'T': b['T'],
                    'absorbed fraction in layer': (sum(b['A']) if j is None else b['A'][j])}[metric]
        self.sweep_worker = SweepWorker(s, self._setter(name), vals, self.sw_quality.currentIndex() == 0)
        def got(v, d):
            self.sw_x.append(v); self.sw_y.append(extract(d)); self.sw_curve.setData(self.sw_x, self.sw_y)
            self.sw_plot.getPlotItem().autoRange(padding=0.03)
            self.sw_plot.getViewBox().forget_home()
            self.sw_plot.getViewBox().remember_home()
            self.status_label.setText(f'Sweep {len(self.sw_x)}/{len(vals)}')
        self.sweep_worker.point.connect(got)
        self.sweep_worker.failed.connect(self.failed)
        self.sweep_worker.finished_all.connect(lambda: self.status_label.setText('Sweep complete'))
        self.sweep_worker.start()

    # ------------------------------------------------------------------ scene state / layout
    def _update_scene_modified(self):
        if self._saved_scene_json is None:
            return
        self._dirty = (self.scene.to_json() != self._saved_scene_json)
        self.setWindowModified(self._dirty)
        star = ' *' if self._dirty else ''
        self.setWindowTitle(f'MultiLayerFocus {__version__}{star}')

    def _mark_scene_saved(self):
        self._saved_scene_json = self.read_scene().to_json()
        self._update_scene_modified()

    def _confirm_discard(self):
        if not self._dirty:
            return True
        reply = QMessageBox.question(
            self, 'Unsaved scene changes',
            'The current scene has unsaved parameter changes. Discard them?',
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return reply == QMessageBox.StandardButton.Discard

    def _update_calculation_summary(self):
        r = self.run_result
        if r is None:
            self.status_label.setText('Last calculation: none')
            self.status_label.setToolTip('Last calculation: none')
            return
        mode = 'Live preview' if r.preview else 'Full Compute'
        stale = (self._last_run_scene_json is not None and
                 self.scene.to_json() != self._last_run_scene_json)
        note = '  |  Parameters changed since result' if stale else ''
        volume = '' if r.volume is None else f'  |  XYZ {len(r.vx)}×{len(r.vy)}×{len(r.vz)}'
        summary = (f'{mode}  |  {r.method}  |  Nθ={r.ntheta}, Nφ={r.nphi}'
                   f'  |  XY {len(r.x)}×{len(r.y)}, Z {len(r.z)}'
                   f'{volume}  |  {r.seconds:.2f} s{note}')
        self.status_label.setText(summary)
        self.status_label.setToolTip(summary)

    def _restore_window_layout(self):
        # Apply the 20/80 split AFTER Qt has laid out and shown the window.
        # Do not restore any splitter settings from earlier upgrades; those
        # caused the sidebar to open at the wrong width on other computers.
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is not None:
            area = screen.availableGeometry()
            requested_w = min(round(area.width() * 0.97), area.width())
            requested_h = min(round(area.height() * 0.95), area.height())
            self.resize(requested_w, requested_h)
            self.move(area.x() + (area.width() - self.width()) // 2,
                      area.y() + (area.height() - self.height()) // 2)
        self._apply_initial_split_ratio()
        # The first layout pass can occur after this function, so reapply
        # once, but never on subsequent user-initiated resizes.
        QTimer.singleShot(150, self._apply_initial_split_ratio)
        self._adapt_header()


    def _apply_initial_split_ratio(self):
        usable = max(1, self.splitter.width() - self.splitter.handleWidth())
        side = round(usable * 0.25)
        self.splitter.setSizes([side, usable - side])

    def _adapt_header(self):
        # Same one-row header; compact labels only if the window is narrow.
        width = self.width()
        self.brand_subtitle.setVisible(width >= 1050)
        self.brand_title.setText('MultiLayerFocus' if width >= 870 else 'MLF')
        self.live.setText('● Live preview' if width >= 980 else '● Preview')
        self.compute_btn.setText('Compute')

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'brand_subtitle'):
            self._adapt_header()

    # ------------------------------------------------------------------ export
    def _need_result(self):
        if self.run_result is None:
            QMessageBox.information(self, 'Nothing to export', 'Run a simulation first.'); return False
        return True

    def export_all_tabs(self):
        from .tab_export import collect_tab, write_tab, safe_name
        if not self._need_result(): return
        folder = QFileDialog.getExistingDirectory(self, 'Choose folder for complete export')
        if not folder: return
        name, ok = QInputDialog.getText(self, 'Export folder name', 'Folder name:', text='MultiLayerFocus_Complete_Export')
        if not ok or not name.strip(): return
        root = Path(folder) / safe_name(name)
        n = 2
        while root.exists():
            root = Path(folder) / f'{safe_name(name)}_{n}'
            n += 1
        root.mkdir()
        old_tab = self.tabs.currentIndex()
        r = self.run_result
        exported, skipped = [], []
        try:
            for i in range(self.tabs.count()):
                self.tabs.setCurrentIndex(i)
                title = self.tabs.tabText(i)
                if i == 7 and not getattr(self, 'sw_x', []):
                    skipped.append(title)
                    continue
                datasets = collect_tab(self)
                if not datasets:
                    skipped.append(title)
                    continue
                source = (getattr(self, '_sweep_scene', None) if i == 7 else r.scene) or self.scene
                metadata = {
                    'app_version': __version__,
                    'scene': json.loads(source.to_json())['scene'],
                    'scene_source': 'parameter sweep' if i == 7 else 'last plotted computation',
                    'results_differ_from_current_parameters': source.to_json() != self.scene.to_json(),
                    'view_controls': self._export_view_controls()
                }
                if i != 7:
                    metadata['calculation'] = {
                        'mode': 'preview' if r.preview else 'full',
                        'method': r.method, 'ntheta': r.ntheta, 'nphi': r.nphi,
                        'xy_samples': [len(r.y), len(r.x)], 'z_samples': len(r.z),
                        'seconds': r.seconds,
                        'sampling_error': float(r.sampling_error) if r.sampling_error is not None and np.isfinite(r.sampling_error) else None
                    }
                else:
                    metadata['sweep'] = dict(getattr(self, '_sweep_export_details', {}))
                    metadata['sweep']['points_exported'] = len(self.sw_x)
                volume = (r.volume, r.vx, r.vy, r.vz) if i == 6 and r.volume is not None else None
                tab_folder = root / safe_name(title)
                tab_folder.mkdir()
                for fmt in ('txt', 'mat'):
                    write_tab(tab_folder, title, datasets, fmt, metadata, volume=volume, folder_name=fmt.upper())
                exported.append(title)
        except Exception as exc:
            traceback.print_exc()
            QMessageBox.critical(self, 'Complete export failed', f'{exc}\n\nPartial output: {root}')
            return
        finally:
            self.tabs.setCurrentIndex(old_tab)
        self.status_label.setText(f'Exported {len(exported)} tabs to {root}')
        QMessageBox.information(self, 'Export completed', f'Data saved successfully!\n\nExported tabs: {len(exported)}\nSkipped tabs: {", ".join(skipped) if skipped else "none"}\n\nLocation:\n{root}')


    def export_tab_txt(self):
        self._export_tab_data('txt')

    def export_tab_mat(self):
        self._export_tab_data('mat')

    def _export_tab_data(self, fmt):
        from .tab_export import collect_tab, write_tab
        try:
            datasets = collect_tab(self)
        except Exception as exc:
            QMessageBox.critical(self, 'Tab data export', f'Cannot read plotted data: {exc}')
            return
        if not datasets:
            QMessageBox.information(self, 'Tab data export',
                                    'No numerical plots are available in this tab yet. '
                                    'Run a simulation or parameter sweep first.')
            return
        folder = QFileDialog.getExistingDirectory(self, 'Choose folder for tab data export')
        if not folder:
            return
        from .tab_export import safe_name
        default_name = f'{safe_name(self.tabs.tabText(self.tabs.currentIndex()))}_{fmt.upper()}'
        name, ok = QInputDialog.getText(self, 'Export folder name', 'Folder name:', text=default_name)
        if not ok or not name.strip(): return
        name = safe_name(name)        
        r = self.run_result
        # Metadata must describe the computation producing these plots, not
        # unsaved or subsequently modified GUI parameters.
        source = (getattr(self, '_sweep_scene', None) if self.tabs.currentIndex() == 7
                  else (r.scene if r is not None else self.scene)) or self.scene
        metadata = {
            'app_version': __version__,
            'scene': json.loads(source.to_json())['scene'],
            'scene_source': 'parameter sweep' if self.tabs.currentIndex() == 7
                            else 'last plotted computation',
            'results_differ_from_current_parameters': (
                source.to_json() != self.scene.to_json()
            ),
            'view_controls': self._export_view_controls(),
        }
        if r is not None and self.tabs.currentIndex() != 7:
            metadata['calculation'] = {
                'mode': 'preview' if r.preview else 'full',
                'method': r.method, 'ntheta': r.ntheta, 'nphi': r.nphi,
                'xy_samples': [len(r.y), len(r.x)],
                'z_samples': len(r.z), 'seconds': r.seconds,
                'sampling_error': (
                    float(r.sampling_error)
                    if r.sampling_error is not None and np.isfinite(r.sampling_error)
                    else None
                ),
            }
        if self.tabs.currentIndex() == 7:
            metadata['sweep'] = dict(getattr(self, '_sweep_export_details', {}))
            metadata['sweep']['points_exported'] = (
                len(self.sw_x) if hasattr(self, 'sw_x') else 0
            )
        try:
            volume = (r.volume, r.vx, r.vy, r.vz) if self.tabs.currentIndex() == 6 and r is not None and r.volume is not None else None
            path = write_tab(folder, self.tabs.tabText(self.tabs.currentIndex()), datasets, fmt, metadata, volume=volume, folder_name=name)
            QMessageBox.information(self, 'Export completed', f'Data saved successfully!\n\nLocation:\n{path}')
        except Exception as exc:
            traceback.print_exc()
            QMessageBox.critical(self, 'Tab data export failed', str(exc))
            return
        self.status_label.setText(f'Exported {len(datasets)} datasets to {path}')

    def _export_view_controls(self):
        tab = self.tabs.currentIndex()
        controls = {'tab_index': tab}
        if tab == 2:
            controls.update(quantity=self.qty.currentText(),
                            axial_section=self.focus_axial_plane.currentText(),
                            normalization=self.norm.currentText(),
                            log_colour_scale=self.logscale.isChecked(),
                            shared_scale=self.shared.isChecked())
        elif tab == 3:
            controls['shared_intensity_scale'] = self.comp_shared.isChecked()
        elif tab == 4:
            controls.update(ellipse_density=self.pol_ellipse_density.value(),
                            ellipse_size_percent=self.pol_ellipse_size.value())
        elif tab == 6:
            controls.update(axial_section=self.vol_axial_plane.currentText(),
                            slice_z_um=self.vol_slider.value())
        return controls

    def export_view_png(self):
        if not self._need_result(): return
        path, _ = QFileDialog.getSaveFileName(self, 'Save view', 'multilayerfocus_view.png', 'PNG (*.png)')
        if path:
            pm = self.tabs.currentWidget().grab(); pm.save(path); self.status_label.setText(f'Saved {path}')

    def export_publication(self):
        if not self._need_result(): return
        path, _ = QFileDialog.getSaveFileName(self, 'Save figure', 'focus_xz_xy.png', 'PNG (*.png);;SVG (*.svg);;PDF (*.pdf)')
        if not path: return
        from .figures import publication_figure
        publication_figure(self.run_result, path, cmap=self.cmap.currentText())
        self.status_label.setText(f'Saved {path}')

    def _arrays(self):
        r = self.run_result
        d = dict(x_um=r.x, y_um=r.y, z_um=r.z, E_xy_Vpm=r.Exy*r.scale, E_xz_Vpm=r.Exz*r.scale, E_yz_Vpm=r.Eyz*r.scale,
                 scene_json=np.asarray(r.scene.to_json()), power_mW=r.scene.power_mW,
                 ntheta=r.ntheta, nphi=r.nphi)
        if r.Hxy is not None:
            d.update(Z0H_xy_Vpm=r.Hxy*r.scale, Z0H_xz_Vpm=r.Hxz*r.scale)
        if r.volume is not None:
            d.update(E_volume_Vpm=r.volume*r.scale, vx_um=r.vx, vy_um=r.vy, vz_um=r.vz)
        return d

    def export_npz(self):
        if not self._need_result(): return
        path, _ = QFileDialog.getSaveFileName(self, 'Save data', 'multilayerfocus_fields.npz', 'NumPy (*.npz)')
        if path: np.savez_compressed(path, **self._arrays()); self.status_label.setText(f'Saved {path}')

    def export_mat(self):
        if not self._need_result(): return
        path, _ = QFileDialog.getSaveFileName(self, 'Save data', 'multilayerfocus_fields.mat', 'MATLAB (*.mat)')
        if path:
            from scipy.io import savemat
            d = self._arrays(); d['scene_json'] = str(d['scene_json']); savemat(path, d, do_compression=True)
            self.status_label.setText(f'Saved {path}')

    def export_csv(self):
        if not self._need_result(): return
        folder = QFileDialog.getExistingDirectory(self, 'Folder for CSV files')
        if not folder: return
        r = self.run_result
        for plane, E, a, b in (('xy', r.Exy, r.x, r.y), ('xz', r.Exz, r.x, r.z)):
            A, B = np.meshgrid(a, b); E = E*r.scale
            cols = [A.ravel(), B.ravel()] + [f(E[c]).ravel() for c in range(3) for f in (np.real, np.imag)]
            hdr = f'{"x" if plane else ""}_um,{"y" if plane == "xy" else "z"}_um,ReEx,ImEx,ReEy,ImEy,ReEz,ImEz (V/m for {r.scene.power_mW} [mW])'
            np.savetxt(os.path.join(folder, f'field_{plane}.csv'), np.column_stack(cols), delimiter=',', header=hdr)
        Path(folder, 'scene.json').write_text(r.scene.to_json(), encoding='utf-8')
        self.status_label.setText(f'Saved CSV files in {folder}')



    def save_scene(self):
        initial = self._scene_path or 'scene.mlf.json'
        path, _ = QFileDialog.getSaveFileName(self, 'Save scene', initial, 'Scene (*.json)')
        if not path:
            return
        try:
            Path(path).write_text(self.read_scene().to_json(), encoding='utf-8')
        except OSError as exc:
            QMessageBox.critical(self, 'Could not save scene', str(exc))
            return
        self._scene_path = path
        self._mark_scene_saved()
        self.status_label.setText(f'Saved {path}')

    def open_scene(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open scene', '', 'Scene (*.json)')
        if not path or not self._confirm_discard():
            return
        try:
            loaded = Scene.from_json(Path(path).read_text(encoding='utf-8'))
            self.set_scene(loaded)
        except Exception as exc:
            QMessageBox.critical(self, 'Could not open scene', str(exc))
            return
        self._scene_path = path
        self._mark_scene_saved()


    def export_report(self):
        if not self._need_result():
            return

        if self.run_result.preview:
            QMessageBox.information(
                self,
                'Detailed simulation report',
                'Run a full Compute before exporting the report. '
                'Live preview uses reduced numerical sampling.'
            )
            return

        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(
                self,
                'Detailed simulation report',
                'Finish the current computation before exporting.'
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            'Save detailed simulation report',
            'multilayerfocus_report.pdf',
            'PDF documents (*.pdf)'
        )

        if not path:
            return

        try:
            from .report_gui import export_detailed_report
            export_detailed_report(self, path)

        except Exception as exc:
            traceback.print_exc()
            QMessageBox.critical(
                self,
                'Report export failed',
                str(exc)
            )
        else:
            self.status_label.setText(
                f'Detailed report saved: {path}'
            )



    def about(self):
        p = PALETTES[self.theme]

        dialog = QMessageBox(self)
        dialog.setWindowTitle('About MultiLayerFocus')
        dialog.setIcon(QMessageBox.Icon.Information)
        dialog.setTextFormat(Qt.TextFormat.RichText)

        dialog.setText(f"""
        <div style="color: {p['text']};">
            <h2>MultiLayerFocus {__version__}</h2>
            <p><b>Vectorial focusing through stratified optical media</b></p>

            <p>
            MultiLayerFocus is a scientific simulation application
            for tightly focused electromagnetic fields in
            homogeneous and multilayer optical systems.
            </p>

            <p>
            The numerical model combines vectorial Debye–Wolf
            diffraction with a multilayer optical response
            based on stable transfer-matrix calculations.
            Depending on the pupil field, the solver uses
            Bessel-based azimuthal integration or
            two-dimensional angular quadrature.
            </p>

            <p><b>Main capabilities</b></p>
            <ul>
                <li>Objective, illumination and material-stack configuration</li>
                <li>Generated and imported complex pupil fields</li>
                <li>Vectorial focal-field reconstruction in XY, XZ and YZ</li>
                <li>Electric-field components and polarization analysis</li>
                <li>Fresnel coefficients and multilayer energy response</li>
                <li>Poynting-vector and absorption analysis</li>
                <li>Optional three-dimensional field reconstruction</li>
                <li>Parameter sweeps and scientific data visualization</li>
                <li>Scene persistence and simulation reports</li>
            </ul>

            <p><b>Numerical accuracy</b></p>
            <p>
            Accuracy depends on angular sampling, spatial sampling,
            material optical constants and the supplied pupil field.
            Users should verify numerical convergence when
            interpreting or publishing quantitative results.
            </p>

            <p><b>Scientific background</b></p>
            <p>
            Based on MultiLayerFocus developed by F. Reda et. al (University of Naples - Federico II). Please cite the associated
            scientific publication when using results in research.
            </p>
        </div>
        """)

        dialog.setStyleSheet(f"""
            QMessageBox {{
                background-color: {p['surface']};
            }}
            QMessageBox QLabel {{
                color: {p['text']};
                background-color: transparent;
            }}
        """)

        dialog.exec()


    def closeEvent(self, e):
        if not self._confirm_discard():
            e.ignore()
            return
        # Do not persist the splitter ratio: every launch starts at 30/70.
        # Users can still drag it freely for the entire current session.
        for w in (self.worker, self.sweep_worker):
            if w is not None and w.isRunning():
                w.stop(); w.wait(3000)
        if self.volume_panel:
            try: self.volume_panel.shutdown()
            except Exception: pass
        super().closeEvent(e)


def main():
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName('MultiLayerFocus')
    if ICON.is_file(): app.setWindowIcon(QIcon(str(ICON)))
    w = MainWindow(); w.show()
    if '--selftest' in sys.argv:
        out = sys.argv[sys.argv.index('--selftest') + 1]
        QTimer.singleShot(500, lambda: _selftest(app, w, out))
    return app.exec()


def _selftest(app, w, out):
    """Run one full computation per example scene, save screenshots and metrics, then quit (build check)."""
    import time
    os.makedirs(out, exist_ok=True)
    report = {}
    def idle(timeout=600):
        t = time.time()
        while time.time() - t < timeout:
            app.processEvents()
            if not ((w.worker is not None and w.worker.isRunning()) or w.preview_timer.isActive() or w.pending):
                break
            time.sleep(0.05)
        for _ in range(10): app.processEvents(); time.sleep(0.02)
    try:
        for k, name in enumerate(['Default configuration']):
            idle(); w.start(preview=False); idle()
            r = w.run_result
            report[name] = dict(seconds=r.seconds, error=r.sampling_error, method=r.method, R=r.budget['R'], T=r.budget['T'],
                                peak_z=r.metrics['z_peak'])
            for i in range(w.tabs.count()):
                w.tabs.setCurrentIndex(i); idle()
            w.tabs.setCurrentIndex(0); idle(); w.grab().save(os.path.join(out, f'scene{k}.png'))

        report['status'] = 'ok'
    except Exception as exc:
        report['status'] = f'failed: {exc!r}'
    Path(out, 'selftest.json').write_text(json.dumps(report, indent=1, default=float), encoding='utf-8')
    app.quit()


if __name__ == '__main__':
    sys.exit(main())
