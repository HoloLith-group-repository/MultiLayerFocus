"""PyQt6 desktop interface. Run with python -m stratified_focus.gui."""
import sys
import shutil
import tempfile
from time import perf_counter
from io import BytesIO
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PyQt6.QtCore import Qt, QEvent, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QBrush, QIcon
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QGroupBox, QAbstractSpinBox, QDoubleSpinBox, QSpinBox, QComboBox, QPushButton,
    QCheckBox, QTableWidget, QTableWidgetItem, QLabel, QTabWidget, QHeaderView, QSizePolicy,
    QFileDialog, QMessageBox, QProgressBar, QScrollArea, QLineEdit)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
from .multilayer import gff_multilayer_response
from .model import Objective, Illumination, Stack, Sampling, SimulationConfig
from .field import simulate, compute_additional, SimulationCancelled
from .analysis import intensity
from .pupil import load_inc_field, generate_inc_field
from .session import save_result, load_result
from .report import build_report
from .volume_gui import VolumePanel
from . import __version__
from .display import WARNING_STYLE, calculation_warning, preview_matches_result
from .numeric_inputs import DecimalSpinBox, parse_decimal


@dataclass
class InputPreview:
    config: SimulationConfig
    pupil_ex: np.ndarray
    pupil_ey: np.ndarray
    pupil_x: np.ndarray
    pupil_y: np.ndarray


def prepare_preview(config, paths=None):
    """Prepare input figures without evaluating any focal fields."""
    config.validate()
    g = config.sampling
    pupil = (load_inc_field(*paths, config.objective, g.phi, g.theta) if paths else
             generate_inc_field(config.objective, config.illumination, g.phi, g.theta))
    return InputPreview(config, *pupil)


class Worker(QThread):
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()
    advanced = pyqtSignal(int)
    stage_changed = pyqtSignal(str,int,int,float)

    def __init__(self, config, paths=None, preview=False, result=None, additional=None):
        super().__init__()
        self.config,self.paths,self.preview = config,paths,preview
        self.result,self.additional = result,additional
        self._last_stage = None; self._last_stage_time = 0.; self._last_percent = None

    def _stage(self, name, done, total):
        now = perf_counter()
        if name != self._last_stage or done in (0,total) or now-self._last_stage_time >= .05:
            self.stage_changed.emit(name,done,total,now-self.started_at)
            self._last_stage = name; self._last_stage_time = now

    def _progress(self, fraction):
        percent = round(100*fraction)
        if percent != self._last_percent:
            self.advanced.emit(percent); self._last_percent = percent

    def run(self):
        self.started_at = perf_counter()
        cancel = self.isInterruptionRequested
        try:
            if cancel(): raise SimulationCancelled()
            if self.preview:
                self._stage('Preparing input preview',0,1)
                result = prepare_preview(self.config,self.paths)
            elif self.additional:
                result = compute_additional(self.result,axial=self.additional == 'axial',
                    volume=self.additional == 'volume',progress=self._progress,
                    stage_progress=self._stage,cancel=cancel)
            else:
                pupil = None
                if self.paths:
                    self._stage('Loading measured pupil',0,1)
                    g = self.config.sampling
                    pupil = load_inc_field(*self.paths,self.config.objective,g.phi,g.theta)
                result = simulate(self.config,pupil,self._progress,
                                  stage_progress=self._stage,cancel=cancel)
            if cancel(): raise SimulationCancelled()
            self.completed.emit(result)
        except SimulationCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(f'{type(exc).__name__}: {exc}')


class Plot(FigureCanvasQTAgg):
    """Render and export one group of simulation figures per tab."""

    def __init__(self):
        self.figure = Figure(figsize=(11, 8), layout='constrained', facecolor='white')
        super().__init__(self.figure)
        with np.load(Path(__file__).with_name('colormaps.npz')) as maps:
            self.default_colors = tuple(ListedColormap(maps[key], name=key)
                                       for key in ('IntColor', 'PhaseColor', 'DivColor'))
        self.set_palette('Turbo')
        self.shared_component_scale = True
        self.panels = []
        self._rendered_result = None
        self._rendered_signature = None

    def invalidate_cache(self):
        self._rendered_result = None
        self._rendered_signature = None

    def set_palette(self, name):
        from matplotlib import colormaps
        self.int_color = {
            'Turbo': colormaps['turbo'],
            'Inferno': colormaps['inferno'],
            'Viridis': colormaps['viridis'],
            'Grayscale': colormaps['gray'],
        }[name]
        self.phase_color = self.default_colors[1]
        self.div_color = self.default_colors[2]

    @staticmethod
    def _stack_position(stack, z):
        """Map an axial coordinate into equal-height schematic medium bands."""
        count = len(stack.indices)
        scale = max(sum(stack.thicknesses[1:-1]), 1.0)
        if count == 1:
            return .5 + np.arctan(z / scale) / np.pi
        if z <= stack.z_in:
            return 1 - 2*np.arctan((stack.z_in-z) / scale) / np.pi
        if z >= stack.z_out:
            return count-1 + 2*np.arctan((z-stack.z_out) / scale) / np.pi
        j = int(np.searchsorted(stack.interfaces, z, side='right'))
        return j + (z-stack.interfaces[j-1]) / stack.thicknesses[j]

    @staticmethod
    def _draw_stack(ax, stack, z0):
        """Draw ordered media with labelled boundaries (schematic, not to scale)."""
        from matplotlib.patches import Rectangle
        from matplotlib import colormaps
        count = len(stack.indices)
        for j, (n, thickness) in enumerate(zip(stack.indices, stack.thicknesses)):
            ax.add_patch(Rectangle((0, j), 1, 1, facecolor=colormaps['Pastel1'](j % 9),
                                   edgecolor='#65758b', linewidth=1))
            role = ('Homogeneous medium' if count == 1 else 'Incident medium' if j == 0
                    else 'Exit medium' if j == count-1 else f'Layer {j+1}')
            d = '∞' if np.isinf(thickness) else f'{thickness:g} [µm]'
            ax.text(.5, j+.5, f'{role}\nn = {n.real:g} + {n.imag:g}i\nd = {d}',
                    ha='center', va='center', fontsize=max(6, min(9, 70/count)),
                    bbox=dict(facecolor='white', alpha=.65, edgecolor='none', pad=2))
        if count > 1:
            for j, z in enumerate(stack.interfaces, start=1):
                ax.text(-.08, j, f'{z:g} [µm]', ha='right', va='center', fontsize=8)
        position = Plot._stack_position(stack, z0)
        ax.plot([-.02, 1.02], [position, position], '--', color='#007eac', lw=2,
                marker='>', markevery=[0, 1], zorder=5)
        ax.annotate(f'XY plane\nz₀ = {z0:g} [µm]', xy=(1.02, position),
                    xytext=(1.08, position), va='center', fontsize=9, color='#00678c',
                    bbox=dict(facecolor='white', alpha=.9, edgecolor='none', pad=2))
        labels = [f'{z:g} [µm]' for z in stack.interfaces] if count > 1 else []
        axis_x = -max(.65, .09*max((len(label) for label in labels), default=0) + .12)
        ax.annotate('+z', xy=(axis_x, count-.1), xytext=(axis_x, .1),
                    arrowprops=dict(arrowstyle='->', color='#1769aa'), ha='center')
        ax.set(xlim=(axis_x-.12, 1.9), ylim=(-.15, count+.15),
               title='Layer stack\nSchematic, not to scale')
        ax.text(.5, -.03, 'Interface labels use the simulation z coordinates',
                transform=ax.transAxes, ha='center', va='top', fontsize=8, wrap=True)
        ax.set_axis_off()


    @staticmethod
    def _draw_map(figure, ax, x, y, values, title, cmap, low, high, label,
                  polar=False, match_colorbar_height=False):
        if polar:
            artist = ax.pcolormesh(x, y, values, shading='gouraud', cmap=cmap,
                                   vmin=low, vmax=high, rasterized=True)
        else:
            artist = ax.imshow(values, cmap=cmap, vmin=low, vmax=high,
                               extent=(x[0], x[-1], min(y), max(y)),
                               origin='upper' if y[0] > y[-1] else 'lower',
                               interpolation='nearest', aspect='equal')
        ax.set(title=title, xlabel='x [mm]' if polar else 'x [µm]',
               ylabel='y [mm]' if polar else 'y [µm]')
        ax.set_aspect('equal')
        ax.tick_params(labelsize=8)
        ax.title.set_fontsize(10)
        if match_colorbar_height:
            # An inset shares the image axes' height even when equal aspect
            # shortens the image inside a wide subplot slot.
            cax = ax.inset_axes((1.045, 0, .065, 1), transform=ax.transAxes)
            cb = figure.colorbar(artist, cax=cax)
        else:
            cb = figure.colorbar(artist, ax=ax, shrink=.82, pad=.02)
        cb.set_label(label, fontsize=9)
        cb.ax.tick_params(labelsize=8)
        if label == 'Phase [rad]':
            cb.set_ticks([-np.pi, 0, np.pi], labels=['−π', '0', 'π'])
        return artist

    def _map(self, ax, x, y, values, title, cmap, low, high, label,
             polar=False, match_colorbar_height=False):
        self.panels.append(dict(kind='map', x=np.asarray(x), y=np.asarray(y),
                                values=np.asarray(values), title=title, cmap=cmap,
                                low=low, high=high, label=label, polar=polar,
                                match_colorbar_height=match_colorbar_height))
        return self._draw_map(self.figure, ax, x, y, values, title, cmap,
                              low, high, label, polar, match_colorbar_height)

    @staticmethod
    def export_panel(panel, image, data_file, dpi):
        """Write the plotted values and their coordinates beside a standalone image."""
        figure = Figure(figsize=(6.5, 5), layout='constrained', facecolor='white')
        ax = figure.subplots()
        if panel['kind'] == 'stack':
            stack = panel['stack']
            Plot._draw_stack(ax, stack, panel['z0'])
            count = len(stack.indices)
            if count == 1:
                starts, ends = [-np.inf], [np.inf]
            else:
                starts = np.r_[-np.inf, stack.interfaces]
                ends = np.r_[stack.interfaces, np.inf]
            table = np.column_stack((np.arange(1, count+1),
                                     np.real(stack.indices), np.imag(stack.indices),
                                     stack.thicknesses, starts, ends))
            columns = 'medium Re_n Im_n thickness_um z_start_um z_end_um'
        elif panel['kind'] == 'line':
            x, s, p = panel['x'], panel['s'], panel['p']
            ax.plot(x, s, label='s', color='#1769aa', lw=1.7)
            ax.plot(x, p, '--', label='p', color='#dd6b20', lw=1.7)
            ax.set(title=panel['title'], xlabel='θ [°]', ylabel=panel['ylabel'],
                   xlim=(0, x[-1]))
            ax.grid(alpha=.25); ax.legend(loc='best', fontsize=9)
            table = np.column_stack((x, s, p))
            columns = 'theta_deg s p'
        else:
            x, y, values = panel['x'], panel['y'], panel['values']
            Plot._draw_map(figure, ax, x, y, values, panel['title'], panel['cmap'],
                           panel['low'], panel['high'], panel['label'], panel['polar'],
                           panel.get('match_colorbar_height', False))
            if panel.get('xlabel'):
                ax.set(xlabel=panel['xlabel'], ylabel=panel['ylabel'])
            if x.ndim == 1 and y.ndim == 1:
                xx, yy = np.meshgrid(x, y)
            else:
                xx, yy = x, y
            table = np.column_stack((xx.ravel(), yy.ravel(), values.ravel()))
            unit = 'mm' if panel['polar'] else 'um'
            columns = f'{panel.get("xname", "x")}_{unit} {panel.get("yname", "y")}_{unit} value'
        plane_note = f'\nXY evaluation plane: z0_um = {panel["z0"]:.17g}' if panel['kind'] == 'stack' else ''
        np.savetxt(data_file, table, header=f'{panel["title"]}{plane_note}\ncolumns: {columns}',
                   fmt='%.17g', encoding='utf-8')
        figure.savefig(image, dpi=dpi, facecolor='white')

    def render(self, name, r):
        signature = (name,self.int_color.name,self.phase_color.name,
                     self.div_color.name,self.shared_component_scale)
        if self._rendered_result is r and self._rendered_signature == signature:
            renderer = getattr(self,'renderer',None)
            # A high-DPI export may leave a differently sized canvas buffer.
            if renderer is None or (renderer.width,renderer.height) != tuple(self.figure.bbox.size):
                self.draw_idle()
            return
        self.figure.clear()
        self.figure.set_layout_engine('constrained')
        self.panels = []
        if name == '1 · Multilayer geometry':
            # Use a smooth angular grid independent of the integration grid.
            angles = np.linspace(0, r.config.objective.theta_max, 500)
            degrees = np.rad2deg(angles)
            a = gff_multilayer_response(r.config.stack, angles, r.config.illumination.wavelength)
            panels = ((np.abs(a.rs), np.abs(a.rp), 'Reflection amplitude', '|r|'),
                      (np.abs(a.ts), np.abs(a.tp), 'Transmission amplitude', '|t|'))
            grid = self.figure.add_gridspec(2, 2, width_ratios=(1, 1.2))
            axes = [self.figure.add_subplot(grid[row, 1]) for row in range(2)]
            for ax, (s, p, title, ylabel) in zip(axes, panels):
                self.panels.append(dict(kind='line', x=degrees, s=s, p=p,
                                        title=title, ylabel=ylabel))
                ax.plot(degrees, s, label='s', color='#1769aa', lw=1.7)
                ax.plot(degrees, p, '--', label='p', color='#dd6b20', lw=1.7)
                ax.set(title=title, xlabel='θ [°]', ylabel=ylabel,
                       xlim=(0, np.rad2deg(angles[-1])))
                ax.grid(alpha=.25)
                ax.legend(loc='best', fontsize=9)
            self._draw_stack(self.figure.add_subplot(grid[:, 0]), r.config.stack, r.config.sampling.z0)
            self.panels.append(dict(kind='stack', stack=r.config.stack, z0=r.config.sampling.z0,
                                    title='Layer stack and XY evaluation plane'))
            self.figure.suptitle('Multilayer response and geometry')

        elif name == '2 · Pupil field':
            self.figure.set_layout_engine(None)
            self.figure.subplots_adjust(left=.125, right=.92, bottom=.08,
                                        top=.91, wspace=.40, hspace=.35)
            axes = self.figure.subplots(2, 2).flat
            x, y = r.pupil_x/1000, r.pupil_y/1000
            ex, ey = r.pupil_ex, r.pupil_ey
            powers = (np.abs(ex)**2, np.abs(ey)**2)
            for ax, data, title in zip(axes, (*powers, np.angle(ex), np.angle(ey)),
                                       ('|Ex|²', '|Ey|²', 'arg(Ex)', 'arg(Ey)')):
                phase = title.startswith('arg')
                self._map(ax, x, y, data, title,
                          self.phase_color if phase else self.int_color,
                          -np.pi if phase else 0, np.pi if phase else 1,
                          'Phase [rad]' if phase else 'Intensity', polar=True)
            self.figure.suptitle('Pupil field components and phases')

        elif name == '3 · XY transverse intensity':
            total = intensity(r.xy)
            transverse = np.sum(np.abs(r.xy[:2])**2, axis=0)
            high = max(float(total.max()), np.finfo(float).eps)
            for ax, data, title in zip(self.figure.subplots(1, 2),
                                       (total, transverse),
                                       ('Ixy = |Ex|² + |Ey|² + |Ez|²',
                                        'Ixy⊥ = |Ex|² + |Ey|²')):
                self._map(ax, r.x, r.y, data, title, self.int_color, 0, high,
                          'Intensity', match_colorbar_height=True)
            self.figure.suptitle('XY Focal Intensity')

        elif name == '5 · Axial Intensity':
            if r.xz is None or r.yz is None:
                self.figure.text(.5, .5, calculation_warning('Axial intensity','Compute XZ and YZ'),
                                 ha='center', va='center', fontsize=10.5, color='#526477')
            else:
                for ax, field, horizontal, axis_name in zip(
                        self.figure.subplots(1, 2), (r.xz, r.yz), (r.x, r.y), ('x','y')):
                    data = intensity(field)
                    self._map(ax, horizontal, r.z, data,
                              f'I{axis_name}z = |Ex|² + |Ey|² + |Ez|²',
                              self.int_color, 0, max(float(data.max()),np.finfo(float).eps),
                              'Intensity')
                    self.panels[-1].update(xname=axis_name, yname='z',
                                            xlabel=f'{axis_name} [µm]', ylabel='z [µm]')
                    ax.set(xlabel=f'{axis_name} [µm]', ylabel='z [µm]')
            self.figure.suptitle('XZ and YZ focal intensity')

        elif name == '4 · XY transverse components':
            axes = self.figure.subplots(2, 3).flat
            powers = np.abs(r.xy)**2
            high = max(float(intensity(r.xy).max()), np.finfo(float).eps)
            contributions = np.trapezoid(np.trapezoid(powers, r.x, axis=2), r.y, axis=1)
            fractions = 100*contributions/contributions.sum()
            for j, ax in enumerate(axes[:3]):
                self._map(ax, r.x, r.y, powers[j],
                          f'|E{"xyz"[j]}|² ({fractions[j]:.1f}%)',
                          self.int_color, 0,
                          high if self.shared_component_scale else max(float(powers[j].max()), np.finfo(float).eps),
                          'Intensity')
            for j, ax in enumerate(axes[3:]):
                self._map(ax, r.x, r.y, np.angle(r.xy[j]), f'arg(E{"xyz"[j]})',
                          self.phase_color, -np.pi, np.pi, 'Phase [rad]')
            self.figure.suptitle('XY focal field components and phases')

        elif name == '6 · 3D polarization ellipse':
            pol = r.polarization
            ab_max = max(float(np.abs(pol.major).max()),
                         float(np.abs(pol.minor).max()), np.finfo(float).eps)
            n_max = max(float(np.abs(pol.normal).max()), np.finfo(float).eps)
            axes = self.figure.subplots(3, 3)
            for row, vectors in enumerate((pol.major/ab_max, pol.minor/ab_max, pol.normal)):
                for col, ax in enumerate(axes[row]):
                    label = 'ABN'[row] + 'xyz'[col]
                    limit = n_max if row == 2 else 1
                    self._map(ax, r.x, r.y, vectors[col], label,
                              self.div_color, -limit, limit,
                              'Amplitude' if row == 2 else 'Normalized amplitude')
            self.figure.suptitle('Major axis, minor axis, and ellipse normal')

        self.draw_idle()
        self._rendered_result = r
        self._rendered_signature = signature


def spin(value, low, high, decimals=3, step=None):
    w = DecimalSpinBox()
    w.setRange(low, high); w.setDecimals(decimals); w.setValue(value)
    if step is not None: w.setSingleStep(step)
    return w


def integer(value, low=2, high=1000):
    w = QSpinBox(); w.setRange(low, high); w.setValue(value)
    return w


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f'MultiLayerFocus {__version__}')
        icon_path = Path(__file__).with_name('icon.png')
        if not icon_path.is_file():
            icon_path = Path(__file__).parent.parent / 'icon.png'
        if icon_path.is_file():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.resize(1320, 850)
        self.result = None
        self.result_dirty = False
        self.results_stale = False
        self._exporting = False
        self.preview = None
        self.worker = None
        self.loaded_legacy = False
        self._pending_worker_outcome = None
        self.loaded_path = None
        self.loaded_metadata = None
        self.input_paths = ()
        self.setStyleSheet('''
            QMainWindow, QScrollArea { background: #f7f9fc; }
            QGroupBox { font-weight: 600; border: 1px solid #d9e1eb; border-radius: 7px;
                margin-top: 12px; padding: 13px 8px 8px; background: white; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }
            QPushButton { background: #1769aa; color: white; padding: 7px 12px; border: 0; border-radius: 5px; }
            QPushButton:disabled { background: #a9b7c4; }
            QTabWidget::pane { border: 1px solid #d9e1eb; }
        ''')
        root = QWidget(); self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        side = QScrollArea(); side.setWidgetResizable(True); side.setMinimumWidth(350); side.setMaximumWidth(420)
        self.side = side
        panel = QWidget(); controls = QVBoxLayout(panel); side.setWidget(panel)
        left_layout.addWidget(side, 1)
        self._controls(controls)
        self.run_button = QPushButton('Run simulation')
        self.run_button.clicked.connect(self.run)
        run_row = QHBoxLayout(); run_row.addWidget(self.run_button,1)
        self.abort_button = QPushButton('Abort')
        self.abort_button.setEnabled(False)
        self.abort_button.setStyleSheet('QPushButton { background: #b42318; } QPushButton:disabled { background: #a9b7c4; }')
        self.abort_button.clicked.connect(self.abort)
        run_row.addWidget(self.abort_button)
        left_layout.addLayout(run_row)
        layout.addWidget(left, 0)
        right = QWidget(); rv = QVBoxLayout(right); layout.addWidget(right, 1)
        toolbar = QHBoxLayout()
        figure_heading = QLabel('Simulation figures')
        figure_heading.setStyleSheet('font-size: 17px; font-weight: 600;')
        toolbar.addWidget(figure_heading)
        self.palette = QComboBox()
        self.palette.addItems(['Turbo', 'Inferno', 'Viridis', 'Grayscale'])
        self.palette.setToolTip('Intensity colormap. Phase and signed-vector colormaps stay fixed.')
        self.palette.currentTextChanged.connect(self.change_palette)
        toolbar.addWidget(self.palette)
        toolbar.addStretch(1)
        self.load_button = QPushButton('Load simulation…')
        self.load_button.clicked.connect(self.load)
        toolbar.addWidget(self.load_button)
        self.export = QPushButton('Save results…')
        self.export.clicked.connect(self.save)
        self.export.setEnabled(False)
        toolbar.addWidget(self.export)
        self.image_format = QComboBox()
        self.image_format.addItems(['PNG', 'TIFF'])
        self.image_format.setMinimumWidth(80)
        self.image_format.setToolTip('Figure format for Export Session')
        toolbar.addWidget(self.image_format)
        self.dpi = integer(300, 72, 1200)
        self.dpi.setToolTip('Figure resolution in dots per inch')
        self.dpi.setSuffix(' dpi')
        toolbar.addWidget(self.dpi)
        self.export_session_button = QPushButton('Export session…')
        self.export_session_button.clicked.connect(self.export_session)
        self.export_session_button.setEnabled(False)
        toolbar.addWidget(self.export_session_button)
        rv.addLayout(toolbar)
        self.tabs = QTabWidget()
        rv.addWidget(self.tabs, 1)
        self.plots = {}
        for name in ('1 · Multilayer geometry', '2 · Pupil field', '3 · XY transverse intensity',
                     '4 · XY transverse components', '5 · Axial Intensity', '6 · 3D polarization ellipse'):
            plot = Plot()
            plot.set_palette(self.palette.currentText())
            self.plots[name] = plot
            page = QWidget(); page_layout = QVBoxLayout(page)
            if name == '4 · XY transverse components':
                self.component_scale = QCheckBox('Shared scale across components')
                self.component_scale.setChecked(True)
                self.component_scale.setToolTip('Uncheck to scale each component from zero to its own maximum')
                self.component_scale.toggled.connect(self.change_component_scale)
                page_layout.addWidget(self.component_scale)
            page_layout.addWidget(plot, 1)
            if name == '5 · Axial Intensity':
                self.axial_warning = QLabel(calculation_warning('Axial intensity','Compute XZ and YZ'))
                self.axial_warning.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.axial_warning.setWordWrap(True); self.axial_warning.setStyleSheet(WARNING_STYLE)
                self.axial_placeholder = QWidget(); empty_layout = QVBoxLayout(self.axial_placeholder)
                empty_layout.addStretch(1); empty_layout.addWidget(self.axial_warning)
                self.compute_axial_button = QPushButton('Compute XZ and YZ')
                self.compute_axial_button.clicked.connect(lambda:self.calculate_additional('axial'))
                empty_layout.addWidget(self.compute_axial_button,0,Qt.AlignmentFlag.AlignHCenter)
                empty_layout.addStretch(1)
                self.axial_placeholder.hide(); page_layout.addWidget(self.axial_placeholder,1)
            self.tabs.addTab(page, name)
        self.tabs.currentChanged.connect(self.refresh)
        self.volume_panel = VolumePanel()
        self.volume_panel.set_palette(self.palette.currentText())
        self.volume_panel.compute_requested.connect(lambda:self.calculate_additional('volume'))
        self.tabs.addTab(self.volume_panel, '7 · XYZ volume')
        self._default_tab_colors = [self.tabs.tabBar().tabTextColor(i) for i in range(self.tabs.count())]
        self.update_additional_buttons()
        self.status = QLabel('Ready · all distances in [µm]')
        self.progress = QProgressBar(); self.progress.setValue(0)
        self.progress.setFormat('Overall %p%')
        self.stale_warning = QLabel()
        self.stale_warning.setWordWrap(True)
        self.stale_warning.setStyleSheet('color: #b42318; font-weight: 700;')
        self.stale_warning.hide()
        rv.addWidget(self.status); rv.addWidget(self.stale_warning); rv.addWidget(self.progress)
        for widget in panel.findChildren(QAbstractSpinBox) + panel.findChildren(QComboBox):
            widget.installEventFilter(self)
            if isinstance(widget, QAbstractSpinBox):
                widget.lineEdit().installEventFilter(self)
        self.dpi.installEventFilter(self)
        self.dpi.lineEdit().installEventFilter(self)
        for name in ('1 · Multilayer geometry', '2 · Pupil field'):
            plot = self.plots[name]
            plot.figure.text(.5, .5, 'Confirm configuration to preview inputs', ha='center', va='center')
            plot.draw_idle()

    def change_component_scale(self, shared):
        self.plots['4 · XY transverse components'].shared_component_scale = shared
        self.refresh()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel and self.side.isAncestorOf(obj):
            QApplication.sendEvent(self.side.viewport(), event)
            return True
        if event.type() == QEvent.Type.Wheel and obj in (self.dpi, self.dpi.lineEdit()):
            return True
        return super().eventFilter(obj, event)

    def change_palette(self, name):
        for plot in self.plots.values():
            plot.set_palette(name)
        if hasattr(self, 'volume_panel'):
            self.volume_panel.set_palette(name)
        self.refresh()

    def _controls(self, layout):
        obj = QGroupBox('Objective and illumination'); f = QFormLayout(obj)
        self.na = spin(.95,.001,1.6,4,.05); f.addRow('Numerical Aperture', self.na)
        self.focal = spin(2620,.001,1e8,2); f.addRow('Focal length [µm]', self.focal)
        self.wave = spin(.532,.3,1.1,4); f.addRow('Vacuum wavelength [µm]', self.wave)
        self.waist = spin(6e6,.001,1e9,1); f.addRow('Pupil waist [µm]', self.waist)
        self.pol = QComboBox(); self.pol.addItems(['uniform','cylindrical']); f.addRow('Polarization', self.pol)
        self.gamma = spin(0,0,90,1); f.addRow('Ex-Ey amplitude γ [deg]', self.gamma)
        self.beta = spin(0,-180,180,1); f.addRow('Ex-Ey phase β [deg]', self.beta)
        self.delta = spin(0,-90,90,1); f.addRow('Radial–Azimuthal angle δ [deg]', self.delta)
        self.vortex = integer(0,-50,50); f.addRow('Vortex charge', self.vortex)
        self.zernike = QLineEdit('0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0')
        self.zernike.setToolTip('15 real Zernike coefficients in radians; n=0…4, m=-n:2:n. Separate coefficients with semicolons when using decimal commas.')
        f.addRow('Zernike coefficients [rad]', self.zernike)
        layout.addWidget(obj)

        external = QGroupBox('Optional measured pupil · complex text files'); ev = QVBoxLayout(external)
        self.source = QCheckBox('Load Cartesian Ex, Ey, x, y'); ev.addWidget(self.source)
        self.measured_fields = QWidget(); ef = QFormLayout(self.measured_fields)
        ev.addWidget(self.measured_fields)
        self.measured_fields.setVisible(False)
        self.source.toggled.connect(self.measured_fields.setVisible)
        self.paths = {}
        for key in ('Ex', 'Ey', 'x', 'y'):
            box = QWidget(); row = QHBoxLayout(box); row.setContentsMargins(0,0,0,0)
            path = QLineEdit(); button = QPushButton('…'); button.setFixedWidth(30)
            button.clicked.connect(lambda _, target=path: self.pick(target))
            row.addWidget(path); row.addWidget(button)
            ef.addRow(key, box); self.paths[key] = path
        layout.addWidget(external)

        layers = QGroupBox('Layer stack · incident → exit')
        self.layer_section = layers; layers.setObjectName('layerStackSection')
        layers.setStyleSheet("""
            QGroupBox#layerStackSection {
                border: 1px solid #d9e1eb;
                border-left: 1px solid #d9e1eb;
                border-right: 1px solid #d9e1eb;
                border-top: 1px solid #d9e1eb;
                border-bottom: 1px solid #d9e1eb;
                border-radius: 7px; margin-top: 12px;
                padding: 13px 6px 8px; background: white;
            }
            QGroupBox#layerStackSection::title {
                subcontrol-origin: margin; left: 12px; padding: 0 4px;
            }
        """)
        v = QVBoxLayout(layers); v.setContentsMargins(0,0,0,0)
        self.objective_band = QLabel('Objective')
        self.objective_band.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.objective_band.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed)
        self.objective_band.setMinimumHeight(36)
        self.objective_band.setStyleSheet('background: #414751; color: white; '
                                        'font-weight: 700; padding: 8px; border: none;')
        v.addWidget(self.objective_band)
        self.layer_table = QTableWidget(3, 3); self.layer_table.setHorizontalHeaderLabels(['Re(n)', 'Im(n)', 'd [µm]'])
        header = self.layer_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.layer_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.layer_table.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed)
        self.layer_table.setStyleSheet('QTableWidget { border: 1px solid #d9e1eb; gridline-color: #d9e1eb; }')
        for heading in (header,self.layer_table.verticalHeader()):
            font = heading.font(); font.setBold(True); heading.setFont(font)
        self.layer_table.setFixedHeight(
            self.layer_table.horizontalHeader().sizeHint().height()
            + 5*self.layer_table.verticalHeader().defaultSectionSize()
            + 2*self.layer_table.frameWidth())
        for row, values in enumerate((('1','0','∞'),('1','0','0.5'),('1','0','∞'))):
            for col, value in enumerate(values):
                self._set_layer_item(row, col, value)
        self._lock_boundary_thickness()
        v.addWidget(self.layer_table)
        self.layer_warning = QLabel()
        self.layer_warning.setWordWrap(True)
        self.layer_warning.setStyleSheet('color: #b42318; font-weight: normal;')
        v.addWidget(self.layer_warning)
        self.layer_table.itemChanged.connect(self.validate_layer_entries)
        self.validate_layer_entries()
        actions = QHBoxLayout()
        add = QPushButton('Add layer'); add.clicked.connect(self.add_layer)
        remove = QPushButton('Remove layer'); remove.clicked.connect(self.remove_layer)
        actions.addWidget(add); actions.addWidget(remove); v.addLayout(actions)
        self.zout = spin(0,-1e6,1e6,3); v.addWidget(QLabel('Last interface z [µm]')); v.addWidget(self.zout)
        layout.addWidget(layers)

        self.preview_button = QPushButton('Confirm configuration / preview inputs')
        self.preview_button.setToolTip('Update the stack and pupil figures without computing focal fields')
        self.preview_button.clicked.connect(self.preview_inputs)
        layout.addWidget(self.preview_button)

        sampling = QGroupBox('Observation and sampling'); sf = QFormLayout(sampling)
        self.z0 = spin(0,-1e5,1e5,3); sf.addRow('Transverse field evaluation z [µm]', self.z0)
        self.x0 = spin(0,-1e5,1e5,3); sf.addRow('Center x [µm]', self.x0)
        self.y0 = spin(0,-1e5,1e5,3); sf.addRow('Center y [µm]', self.y0)
        self.width = spin(2,.001,1e5,3); sf.addRow('Lateral width [µm]', self.width)
        self.height = spin(2,.001,1e5,3); sf.addRow('Axial extent [µm]', self.height)
        self.nphi = integer(200); sf.addRow('Azimuthal samples', self.nphi)
        self.ntheta = integer(200); sf.addRow('Polar samples', self.ntheta)
        self.nxy = integer(100); sf.addRow('XY samples per axis', self.nxy)
        self.nz = integer(100); sf.addRow('Axial z samples', self.nz)
        self.axial = QCheckBox('Compute XZ and YZ'); sf.addRow(self.axial)
        self.compute_volume = QCheckBox('Compute XYZ volume'); sf.addRow(self.compute_volume)
        self.volume_options = QWidget(); vf = QFormLayout(self.volume_options)
        self.volume_xy = integer(41, 2, 501); vf.addRow('Volume XY samples per axis', self.volume_xy)
        self.volume_nz = integer(41, 2, 501); vf.addRow('Volume z samples', self.volume_nz)
        self.volume_memory = QLabel(); self.volume_memory.setWordWrap(True); vf.addRow(self.volume_memory)
        sf.addRow(self.volume_options); self.volume_options.setVisible(False)
        self.compute_volume.toggled.connect(self.volume_options.setVisible)
        for widget in (self.volume_xy, self.volume_nz, self.width, self.height):
            widget.valueChanged.connect(self.update_volume_estimate)
        self.update_volume_estimate()
        layout.addWidget(sampling)

        layout.addStretch(1)

    def update_volume_estimate(self, *_):
        nx, nz = self.volume_xy.value(), self.volume_nz.value()
        self.volume_memory.setText(f'{nx} × {nx} × {nz} voxels\n'
            f'Δx = {self.width.value()/(nx-1):.4g} [µm] · '
            f'Δy = {self.width.value()/(nx-1):.4g} [µm] · '
            f'Δz = {self.height.value()/(nz-1):.4g} [µm]')

    def add_layer(self):
        if self.layer_table.rowCount() == 1:
            row = 1
            self.layer_table.insertRow(row)
            for col, value in enumerate(('1.5', '0', '∞')):
                self._set_layer_item(row, col, value)
            self._lock_boundary_thickness()
            self.layer_table.selectRow(row)
            return
        row = self.layer_table.rowCount()-1
        self.layer_table.insertRow(row)
        for col, value in enumerate(('1.5','0','0.5')):
            self._set_layer_item(row, col, value)
        self._lock_boundary_thickness()
        self.layer_table.selectRow(row)

    def remove_layer(self):
        row = self.layer_table.currentRow()
        count = self.layer_table.rowCount()
        if count == 1:
            return
        if row < 0:
            row = count-1
        if row == 0:
            QMessageBox.information(self, 'Incident medium',
                                    'The first medium is fixed. Select another row to remove it.')
            return
        self.layer_table.removeRow(row)
        self._lock_boundary_thickness()
        self.layer_table.selectRow(min(row, self.layer_table.rowCount()-1))

    def _set_layer_item(self, row, col, value):
        self.layer_table.setItem(row, col, QTableWidgetItem(value))

    def _lock_boundary_thickness(self):
        last = self.layer_table.rowCount()-1
        for row in range(self.layer_table.rowCount()):
            item = self.layer_table.item(row, 2)
            if item is None:
                item = QTableWidgetItem()
                self.layer_table.setItem(row, 2, item)
            boundary = row in (0, last)
            if boundary:
                item.setText('∞')
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable
                          if boundary else item.flags() | Qt.ItemFlag.ItemIsEditable)
            item.setToolTip('Semi-infinite boundary medium; thickness is fixed'
                            if boundary else 'Finite layer thickness [µm]')
        if hasattr(self, 'layer_warning'):
            self.validate_layer_entries()

    def validate_layer_entries(self, *_):
        """Flag invalid cells without changing the value typed by the user."""
        errors = []
        previous = self.layer_table.blockSignals(True)
        try:
            from matplotlib import colormaps
            last = self.layer_table.rowCount()-1
            number_font = self.layer_table.verticalHeader().font(); number_font.setBold(True)
            value_font = self.layer_table.font(); value_font.setBold(False)
            for row in range(last+1):
                color = QColor.fromRgbF(*map(float,colormaps['Pastel1'](row % 9)[:3]))
                number = self.layer_table.verticalHeaderItem(row)
                if number is None:
                    number = QTableWidgetItem(); self.layer_table.setVerticalHeaderItem(row,number)
                number.setText(str(row+1)); number.setFont(number_font)
                number.setBackground(QBrush(color))
                for col in range(3):
                    item = self.layer_table.item(row, col)
                    if item is None:
                        errors.append(f'Layer {row+1}: missing value')
                        continue
                    error = None
                    if col == 2 and row in (0, last):
                        pass
                    else:
                        try:
                            value = parse_decimal(item.text())
                            valid = np.isfinite(value) and (value >= 0 if col == 1 else value > 0)
                        except ValueError:
                            valid = False
                        if not valid:
                            field = ('Re(n)', 'Im(n)', 'thickness')[col]
                            rule = '≥ 0' if col == 1 else '> 0'
                            error = f'Layer {row+1}: {field} must be finite and {rule}'
                            errors.append(error)
                    item.setForeground(QBrush(QColor('#b42318' if error else '#172b4d')))
                    item.setBackground(QBrush(QColor('#fff0ef') if error else color))
                    item.setFont(value_font)
                    item.setToolTip(error or ('Semi-infinite boundary medium; thickness is fixed'
                                             if col == 2 and row in (0, last) else
                                             'Lossless media may use Im(n) = 0' if col == 1 else
                                             'Enter a positive finite value'))
        finally:
            self.layer_table.blockSignals(previous)
        self.layer_warning.setText('\n'.join(errors))
        self.layer_warning.setVisible(bool(errors))
        return errors

    def pick(self, widget):
        file, _ = QFileDialog.getOpenFileName(self, 'Select text data')
        if file: widget.setText(file)

    def config(self):
        errors = self.validate_layer_entries()
        if errors:
            raise ValueError('Correct the highlighted layer entries:\n' + '\n'.join(errors))
        ns, ds = [], []
        for row in range(self.layer_table.rowCount()):
            def cell(col):
                item = self.layer_table.item(row, col)
                return item.text().strip() if item else ''
            ns.append(complex(parse_decimal(cell(0)), parse_decimal(cell(1))))
            ds.append(float('inf') if row in (0,self.layer_table.rowCount()-1) else parse_decimal(cell(2)))
        coefficient_text = self.zernike.text()
        separator = ';' if ';' in coefficient_text else ','
        coefficients = tuple(parse_decimal(part) for part in coefficient_text.split(separator))
        cfg = SimulationConfig(
            Objective(self.na.value(),self.focal.value(),ns[0].real),
            Illumination(self.wave.value(),self.waist.value(),('spiral' if self.pol.currentText() == 'cylindrical' else 'uniform'),
                         self.gamma.value(),self.beta.value(),self.delta.value(),self.vortex.value(),coefficients),
            Stack(tuple(ns),tuple(ds),self.zout.value()),
            Sampling(self.nphi.value(),self.ntheta.value(),self.nxy.value(),self.nz.value(),
                     self.width.value(),self.height.value(),self.x0.value(),self.y0.value(),
                     self.z0.value(),self.axial.isChecked(),self.compute_volume.isChecked(),
                     self.volume_xy.value(),self.volume_nz.value()))
        cfg.validate()
        paths = [self.paths[key].text().strip() for key in ('Ex','Ey','x','y')] if self.source.isChecked() else None
        if paths and not all(Path(p).is_file() for p in paths):
            raise ValueError('Select all four existing Cartesian pupil text files')
        return cfg, paths

    def preview_inputs(self):
        if self.worker is not None:
            return
        try:
            config, paths = self.config()
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, 'Invalid parameters', str(exc)); return
        self._start_worker(Worker(config,paths,preview=True))

    def preview_finished(self, preview):
        try:
            for name in ('1 · Multilayer geometry', '2 · Pupil field'):
                self.plots[name].render(name, preview)
        except Exception as exc:
            self.failed(f'{type(exc).__name__}: {exc}')
            return
        self.preview = preview
        self.preview_button.setEnabled(True)
        self.run_button.setEnabled(True)
        self.load_button.setEnabled(True)
        self.progress.setValue(100)
        self._update_result_markers()
        self.status.setText('Input preview updated' if self.results_stale else
                            'Input preview ready · run a simulation to calculate fields' if self.result is None else
                            'Input preview updated · results match the confirmed inputs')
        if self.tabs.currentIndex() > 1:
            self.tabs.setCurrentIndex(0)

    def run(self):
        if self.worker is not None or self._exporting:
            return
        try: config, paths = self.config()
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, 'Invalid parameters', str(exc)); return
        if not self._confirm_result_replacement('starting a new simulation'):
            return
        self._start_worker(Worker(config,paths))

    def _start_worker(self, worker):
        self.worker = worker; self._pending_worker_outcome = None
        self.progress.setValue(0); self.status.setText('Preparing calculation…')
        self._set_calculation_busy(True)
        worker.advanced.connect(self.progress.setValue)
        worker.stage_changed.connect(self.calculation_progress)
        worker.completed.connect(lambda result:self._record_worker_outcome('completed',result))
        worker.failed.connect(lambda message:self._record_worker_outcome('failed',message))
        worker.cancelled.connect(lambda:self._record_worker_outcome('cancelled',None))
        # Restore controls and commit results only after the thread actually exits.
        worker.finished.connect(self._worker_stopped)
        worker.start()

    def _record_worker_outcome(self, kind, value):
        self._pending_worker_outcome = (kind,value)

    def _worker_stopped(self):
        worker = self.worker
        outcome = self._pending_worker_outcome
        self.worker = None; self._pending_worker_outcome = None
        self._set_calculation_busy(False)
        worker.deleteLater()
        if outcome is None:
            self.failed('Calculation ended without a result'); return
        kind,value = outcome
        if kind == 'completed':
            if worker.preview:
                self.preview_finished(value)
            else:
                if not worker.additional:
                    self.input_paths = tuple(worker.paths or ())
                task_elapsed = perf_counter()-worker.started_at
                self.finished(value,preserve_preview=bool(worker.additional))
                if worker.additional:
                    name = 'XZ and YZ sections' if worker.additional == 'axial' else 'XYZ volume'
                    self.status.setText(f'{name} added in {task_elapsed:.2f} s      ·      total computation {value.elapsed_seconds:.2f} s      ·      normalization {value.normalization:.5g}')
        elif kind == 'cancelled':
            self.progress.setValue(0)
            self.status.setText('Calculation aborted · previous completed result retained' if self.result is not None
                                else 'Calculation aborted · no result was saved')
        else:
            self.failed(value)
        self.update_additional_buttons()

    def _set_calculation_busy(self, busy):
        for button in (self.preview_button,self.run_button,self.load_button):
            button.setEnabled(not busy)
        self.export.setEnabled(not busy and self.result is not None)
        self.export_session_button.setEnabled(not busy and self.result is not None)
        self.side.widget().setEnabled(not busy)
        self.abort_button.setEnabled(busy); self.abort_button.setText('Abort')
        self.update_additional_buttons(busy)

    def update_additional_buttons(self, busy=None):
        if busy is None:
            busy = self.worker is not None
        available = self.result is not None and not self.loaded_legacy and not busy
        self.compute_axial_button.setEnabled(available and (self.result.xz is None or self.result.yz is None))
        self.volume_panel.compute_button.setEnabled(available and self.result.volume is None)
        tip = ('Uses the completed simulation configuration and sampling, including its stored pupil. '
               'Changes in the input panel apply only to a new simulation.' if not self.loaded_legacy else
               'This older snapshot lacks complete inputs. Run a new simulation to calculate additional outputs.')
        self.compute_axial_button.setToolTip(tip); self.volume_panel.compute_button.setToolTip(tip)

    def calculate_additional(self, kind):
        if self.worker is not None: return
        if self.result is None or self.loaded_legacy: return
        if kind not in ('axial','volume'): raise ValueError('Unknown additional calculation')
        self._start_worker(Worker(self.result.config,result=self.result,additional=kind))

    def abort(self):
        if self.worker is None or not self.worker.isRunning(): return
        self.worker.requestInterruption()
        self.abort_button.setEnabled(False); self.abort_button.setText('Stopping…')
        self.status.setText('Stopping… waiting for the current numerical operation')

    def calculation_progress(self, name, done, total, elapsed):
        if self.worker is None or self.worker.isInterruptionRequested(): return
        unit = 'angular steps' if name == 'XY transverse field' else 'axial rows' if name == 'XZ and YZ sections' else 'z planes' if name == 'XYZ volume' else 'steps'
        self.status.setText(f'{name} · {done}/{total} {unit} ({100*done/total:.0f}%) · elapsed {elapsed:.1f} s')

    def _set_result_dirty(self, dirty):
        self.result_dirty = bool(dirty)
        self.setWindowTitle(f'MultiLayerFocus {__version__}' + (' *' if self.result_dirty else ''))

    def _confirm_result_replacement(self, action):
        if self.result is None or not self.result_dirty:
            return True
        choice = QMessageBox.question(self,'Unsaved simulation result',
            f'The current simulation result has not been saved. Save it before {action}?',
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save)
        if choice == QMessageBox.StandardButton.Save:
            return self.save()
        # Retain the dirty flag until replacement succeeds, in case the new task aborts.
        return choice == QMessageBox.StandardButton.Discard

    def _invalidate_plot_caches(self):
        for plot in self.plots.values():
            plot.invalidate_cache()

    def _update_result_markers(self):
        self.results_stale = self.preview is not None and self.result is not None and (
            self.loaded_legacy or not preview_matches_result(self.preview,self.result))
        message = ('Red tabs are not updated: they show results from the last simulation. '
                   'Run a new simulation to use the confirmed inputs.')
        for index in range(2,self.tabs.count()):
            self.tabs.tabBar().setTabTextColor(index,QColor('#b42318') if self.results_stale else self._default_tab_colors[index])
            self.tabs.setTabToolTip(index,message if self.results_stale else '')
        self.stale_warning.setText(message if self.results_stale else '')
        self.stale_warning.setVisible(self.results_stale)

    def finished(self, result, preserve_preview=False):
        self._invalidate_plot_caches()
        self.volume_panel.invalidate()
        self.result = result
        self._set_result_dirty(True)
        if not preserve_preview: self.preview = None
        self.preview_button.setEnabled(True)
        self.load_button.setEnabled(True)
        self.loaded_path = None
        self.loaded_metadata = None
        self.loaded_legacy = False
        self.run_button.setEnabled(True)
        self.export.setEnabled(True)
        self.export_session_button.setEnabled(True)
        self.progress.setValue(100)
        self.status.setText(f'Complete in {result.elapsed_seconds:.2f} [s]      ·      normalization {result.normalization:.5g}      ·      z_in {result.config.stack.z_in:g} [µm]')
        self.update_additional_buttons()
        self._update_result_markers()
        self.refresh()

    def failed(self, message):
        self.export.setEnabled(self.result is not None)
        self.export_session_button.setEnabled(self.result is not None)
        self.preview_button.setEnabled(True)
        self.load_button.setEnabled(True)
        self.run_button.setEnabled(True); self.status.setText('Simulation failed')
        QMessageBox.critical(self, 'Simulation error', message)

    def refresh(self, *_):
        name = self.tabs.tabText(self.tabs.currentIndex())
        data = (self.preview if self.preview is not None else self.result) if self.tabs.currentIndex() < 2 else self.result
        if data is not None:
            try:
                if name == '7 · XYZ volume':
                    self.volume_panel.render(name, data)
                else:
                    if name == '5 · Axial Intensity':
                        unavailable = data.xz is None or data.yz is None
                        self.axial_placeholder.setVisible(unavailable)
                        self.plots[name].setVisible(not unavailable)
                    self.plots[name].render(name, data)
            except Exception as exc:
                QMessageBox.critical(self, 'Plot error', f'{type(exc).__name__}: {exc}')

    def _restore_controls(self, config):
        o, light, stack, g = config.objective, config.illumination, config.stack, config.sampling
        for widget, value in ((self.na,o.na),
                              (self.focal,o.focal_length),(self.wave,light.wavelength),
                              (self.waist,light.waist),(self.gamma,light.gamma_deg),
                              (self.beta,light.beta_deg),(self.delta,light.delta_deg),
                              (self.vortex,light.vortex_charge),(self.zout,stack.z_out),
                              (self.z0,g.z0),(self.x0,g.x0),(self.y0,g.y0),
                              (self.width,g.lateral_range),(self.height,g.axial_range),
                              (self.nphi,g.phi),(self.ntheta,g.theta),
                              (self.nxy,g.xy),(self.nz,g.axial),
                              (self.volume_xy,g.volume_xy),(self.volume_nz,g.volume_z)):
            widget.setValue(value)
        self.pol.setCurrentText('cylindrical' if light.polarization == 'spiral' else light.polarization)
        self.zernike.setText(', '.join(f'{v:g}' for v in light.zernike))
        self.axial.setChecked(g.compute_axial)
        self.compute_volume.setChecked(g.compute_volume)
        self.layer_table.setRowCount(len(stack.indices))
        for row, (n, thickness) in enumerate(zip(stack.indices, stack.thicknesses)):
            values = (f'{n.real:g}', f'{n.imag:g}',
                      '∞' if np.isinf(thickness) else f'{thickness:g}')
            for col, value in enumerate(values):
                self._set_layer_item(row, col, value)
        self._lock_boundary_thickness()

    def load(self):
        if self.worker is not None or self._exporting: return
        path, _ = QFileDialog.getOpenFileName(self, 'Load simulation', '',
                                              'NumPy archives (*.npz)')
        if not path:
            return
        try:
            loaded = load_result(path)
        except Exception as exc:
            QMessageBox.critical(self, 'Load failed', f'{type(exc).__name__}: {exc}')
            return
        if not self._confirm_result_replacement('loading another simulation'):
            return
        self._invalidate_plot_caches()
        self.result = loaded.result
        self._set_result_dirty(False)
        self.volume_panel.invalidate()
        self.preview = None
        self.source.setChecked(False)
        self.loaded_legacy = loaded.legacy
        self.loaded_metadata = loaded.metadata
        self.loaded_path = Path(path)
        self.input_paths = tuple(loaded.metadata.get('source_files',()))
        if not loaded.legacy:
            self._restore_controls(loaded.result.config)
        self.export.setEnabled(True)
        self.export_session_button.setEnabled(True)
        self.progress.setValue(100)
        self.status.setText('Loaded earlier NPZ: plots restored; incomplete inputs remain unknown'
                            if loaded.legacy else f'Loaded {Path(path).name}: plots and inputs restored')
        self.update_additional_buttons()
        self._update_result_markers()
        self.refresh()

    def _save_snapshot(self, path):
        if self.loaded_path is not None:
            if self.loaded_path.resolve() != Path(path).resolve():
                shutil.copy2(self.loaded_path, path)
            return self.loaded_metadata
        return save_result(path, self.result, self.input_paths)

    def save(self):
        if self.result is None:
            return False
        path, _ = QFileDialog.getSaveFileName(self, 'Save simulation',
                                               'focus_results.npz', 'NumPy archives (*.npz)')
        if not path:
            return False
        if not path.lower().endswith('.npz'):
            path += '.npz'
        try:
            self.loaded_metadata = self._save_snapshot(path)
            self._set_result_dirty(False)
            self.status.setText(f'Saved {path}')
            return True
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, 'Save failed', str(exc))
            return False

    def export_session(self):
        if self.result is None or self.worker is not None or self._exporting:
            return
        parent = QFileDialog.getExistingDirectory(self, 'Choose a folder for the exported session')
        if not parent:
            return
        stamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
        destination = Path(parent) / f'MultiLayerFocus_{stamp}'
        suffix = 2
        while destination.exists():
            destination = Path(parent) / f'MultiLayerFocus_{stamp}_{suffix}'
            suffix += 1
        temporary = Path(tempfile.mkdtemp(prefix='.stratified-focus-', dir=parent))
        fmt = self.image_format.currentText().lower()
        figures = []
        panel_names = (
            ('reflection_amplitude', 'transmission_amplitude', 'layer_stack'),
            ('ex_intensity', 'ey_intensity', 'ex_phase', 'ey_phase'),
            ('total_intensity', 'transverse_intensity'),
            ('ex_intensity', 'ey_intensity', 'ez_intensity', 'ex_phase', 'ey_phase', 'ez_phase'),
            ('xz_intensity', 'yz_intensity'),
            ('major_x', 'major_y', 'major_z', 'minor_x', 'minor_y', 'minor_z',
             'normal_x', 'normal_y', 'normal_z'),
        )
        self._exporting = True
        self.update_additional_buttons(busy=True)
        self.export_session_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.preview_button.setEnabled(False)
        self.load_button.setEnabled(False)
        stage = 'saving simulation data'
        original_tab = self.tabs.currentIndex()
        try:
            metadata = self._save_snapshot(temporary / 'simulation.npz')
            for index, (name, plot) in enumerate(self.plots.items(), start=1):
                stage = f'exporting section {index} ({name})'
                plot.render(name, self.result)
                section = temporary / f'{index:02d}_{name.split(" · ", 1)[1].lower().replace(" ", "_")}'
                section.mkdir()
                if plot.panels:
                    for panel, stem in zip(plot.panels, panel_names[index-1], strict=True):
                        Plot.export_panel(panel, section / f'{stem}.{fmt}',
                                          section / f'{stem}.txt', self.dpi.value())
                if plot.panels:
                    overview = BytesIO()
                    plot.figure.savefig(overview, format='png', dpi=150, facecolor='white')
                    overview.seek(0)
                    figures.append(overview)
                else:
                    figures.append(None)
                self.progress.setValue(round(100*index/8))
                self.status.setText(f'Exporting section {index}/7…')
                QApplication.processEvents()
            stage = 'exporting the XYZ volume'
            if self.result.volume is not None:
                self.tabs.setCurrentIndex(6)
                QApplication.processEvents()
            self.status.setText('Exporting volume data and views…')
            def volume_progress(value):
                self.progress.setValue(round(75+12*value))
                QApplication.processEvents()
            figures.append(self.volume_panel.export_session(temporary/'07_xyz_volume', self.result,
                                                           fmt, self.dpi.value(), volume_progress))
            stage = 'building the PDF report'
            build_report(temporary / 'report.pdf', self.result, metadata, figures)
            for overview in figures:
                if overview is not None:
                    overview.close()
            stage = 'finalizing the session folder'
            temporary.rename(destination)
            self._set_result_dirty(False)
            self.progress.setValue(100)
            self.status.setText(f'Exported session: {destination}')
        except Exception as exc:
            for overview in figures:
                if overview is not None:
                    overview.close()
            shutil.rmtree(temporary, ignore_errors=True)
            QMessageBox.critical(self, 'Export failed',
                                 f'While {stage}:\n{type(exc).__name__}: {exc}')
            self.status.setText('Session export failed')
        finally:
            self._exporting = False
            self.export_session_button.setEnabled(True)
            busy = self.worker is not None
            self.run_button.setEnabled(not busy)
            self.preview_button.setEnabled(not busy)
            self.load_button.setEnabled(not busy)
            self.update_additional_buttons(busy)
            self.tabs.setCurrentIndex(original_tab)
            self.refresh()

    def closeEvent(self, event):
        if self._exporting:
            QMessageBox.information(self,'Export in progress','Wait for the session export to finish before closing the application.')
            event.ignore(); return
        if self.worker is not None:
            QMessageBox.information(self, 'Calculation in progress',
                                    'Use Abort and wait for the calculation to stop before closing the application.')
            event.ignore(); return
        if not self._confirm_result_replacement('closing the application'):
            event.ignore(); return
        self.volume_panel.shutdown()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    app.setWindowIcon(window.windowIcon())
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
