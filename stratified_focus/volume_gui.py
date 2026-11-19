"""Interactive volume and XY-slice controls for the desktop interface."""
from io import BytesIO
from pathlib import Path
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (QWidget, QFrame, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QSlider, QSplitter, QCheckBox, QGroupBox, QSizePolicy, QToolButton)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib import colormaps
from .display import WARNING_STYLE, calculation_warning
from .numeric_inputs import DecimalSpinBox, PercentageSpinBox
from .volume import volume_peak, volume_intensity, export_volume_data
from scipy.ndimage import zoom

class VolumePanel(QWidget):
    compute_requested = pyqtSignal()
    def __init__(self):
        super().__init__()
        self.result = None; self.values = None; self.peak = None
        self.int_color = colormaps['turbo']; self.scene = None; self.vtk_widget = None
        self.vtk_initialized = False
        self.vtk_refresh_timer = QTimer(self); self.vtk_refresh_timer.setSingleShot(True)
        self.vtk_refresh_timer.timeout.connect(self._refresh_3d_view)
        self.camera_timer = QTimer(self); self.camera_timer.setSingleShot(True)
        self.camera_timer.setInterval(40); self.camera_timer.timeout.connect(self._apply_camera_controls)
        self.high = 1.
        self.slice_artist = None; self.slice_colorbar = None
        self.slice_timer = QTimer(self); self.slice_timer.setSingleShot(True)
        self.slice_timer.setInterval(40)
        self.slice_timer.timeout.connect(self.draw_slices)
        self.figure = Figure(figsize=(6, 6), layout='constrained')
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QVBoxLayout(self)
        bar = QHBoxLayout()
        bar.addWidget(QLabel('Intensity'))
        self.component = QComboBox(); self.component.addItems(['Total', 'Ex', 'Ey', 'Ez'])
        self.component.setMinimumWidth(90)
        bar.addWidget(self.component)
        self.shared_scale = QCheckBox('Shared Scale Across Components')
        self.shared_scale.setToolTip('Unchecked: use the selected component maximum across the full volume. '
                                    'Checked: use the total-volume maximum for every component.')
        bar.addWidget(self.shared_scale); bar.addStretch(1)
        self.component_controls = QWidget(); self.component_controls.setLayout(bar)
        layout.addWidget(self.component_controls)
        self.component_title = QLabel('Total Intensity')
        self.component_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.component_title.setStyleSheet('font-size: 18px; font-weight: 600;')
        layout.addWidget(self.component_title)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.render_host = QWidget()
        white_palette = self.render_host.palette()
        white_palette.setColor(QPalette.ColorRole.Window,QColor('white'))
        self.render_host.setPalette(white_palette)
        self.render_host.setAutoFillBackground(True)
        self.render_layout = QVBoxLayout(self.render_host)
        self.render_layout.setContentsMargins(4,4,4,4)
        heading = QLabel('3D Volume'); heading.setStyleSheet('font-size: 15px; font-weight: 600;')
        self.render_layout.addWidget(heading)
        tuning = QGroupBox('Visualization Settings')
        tuning_layout = QVBoxLayout(tuning)
        opacity_bar = QHBoxLayout(); opacity_bar.setSpacing(8)
        tuning_layout.addLayout(opacity_bar)
        self.threshold = self._spin(.02, 0, 1, 3, percentage=True)
        self.gamma = self._spin(1.6, .1, 10, 2)
        self.opacity = self._spin(1, 0, 1, 2, percentage=True)
        for index,(name, widget) in enumerate((('Threshold', self.threshold), ('Gamma', self.gamma), ('Opacity', self.opacity))):
            widget.setFixedWidth(max(90, widget.sizeHint().width()))
            if index:
                separator = QLabel('|'); separator.setStyleSheet('color: #7a8794;')
                opacity_bar.addWidget(separator)
            opacity_bar.addWidget(self._setting_pair(name,widget))
        opacity_bar.addStretch(1)
        self.threshold.setToolTip('Hide intensities below this percentage of the current color-scale maximum; display only.')
        self.gamma.setToolTip('Opacity grows as relative intensity raised to gamma. '
                              'Above 1 emphasizes bright regions; below 1 reveals weaker regions. Display only.')
        self.opacity.setToolTip('Maximum opacity at the color-scale maximum; display only.')
        self.reset = QPushButton('Reset View')
        self.reset.setMinimumWidth(100)
        opacity_bar.addWidget(self.reset)
        self.render_layout.addWidget(tuning)
        self.show_plane = QCheckBox('Show XY Slice Plane')
        self.show_plane.setChecked(True)
        overlay_row = QHBoxLayout(); overlay_row.setSpacing(8)
        overlay_row.addWidget(self.show_plane)
        self.plane_opacity = self._spin(.30, 0, 1, 2, percentage=True)
        self.plane_opacity.setSingleStep(.02); self.plane_opacity.setFixedWidth(max(90, self.plane_opacity.sizeHint().width()))
        overlay_row.addWidget(self._setting_pair('Opacity',self.plane_opacity))
        self.show_stack = QCheckBox('Show Layer Stack')
        self.show_stack.setChecked(True)
        self.show_stack.setToolTip('Shows translucent layer slabs within the calculated volume. '
                                  'The semi-infinite end media are clipped to the displayed bounds.')
        stack_row = QHBoxLayout(); stack_row.setSpacing(8); stack_row.addWidget(self.show_stack)
        self.stack_opacity = self._spin(.10,0,1,3, percentage=True)
        self.stack_opacity.setFixedWidth(max(90, self.stack_opacity.sizeHint().width()))
        stack_row.addWidget(self._setting_pair('Opacity',self.stack_opacity))
        overlay_groups = QHBoxLayout(); overlay_groups.setSpacing(14)
        overlay_groups.addLayout(overlay_row); overlay_groups.addStretch(1)
        overlay_groups.addLayout(stack_row)
        tuning_layout.addLayout(overlay_groups)
        self.camera_toggle = QToolButton()
        self.camera_toggle.setObjectName('volumeCameraToggle')
        self.camera_toggle.setText('Camera Controls'); self.camera_toggle.setCheckable(True)
        self.camera_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.camera_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.camera_toggle.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed)
        self.camera_toggle.setMinimumHeight(34)
        self.camera_toggle.toggled.connect(self._toggle_camera_controls)
        tuning_layout.addWidget(self.camera_toggle)
        self.camera_controls = QWidget(); camera_layout = QVBoxLayout(self.camera_controls)
        camera_layout.setContentsMargins(0,0,0,0); camera_layout.setSpacing(6)
        self.camera_entries = []; self.camera_sliders = []
        for index,(label,low,high,initial) in enumerate((('Azimuth',-180,180,-45),('Elevation',-89,89,30))):
            row = QHBoxLayout(); row.setSpacing(4)
            name_label = QLabel(label); name_label.setFixedWidth(70)
            row.addWidget(name_label)
            entry = DecimalSpinBox(); entry.setRange(low,high); entry.setDecimals(1)
            entry.setSuffix(' [°]'); entry.setSingleStep(1); entry.setValue(initial)
            entry.setKeyboardTracking(False); entry.setFixedWidth(max(90,entry.sizeHint().width()))
            slider = QSlider(Qt.Orientation.Horizontal); slider.setRange(low*10,high*10)
            slider.setValue(initial*10); slider.setSingleStep(10)
            entry.valueChanged.connect(lambda value,i=index:self._camera_control_changed(i,value))
            slider.valueChanged.connect(lambda value,i=index:self._camera_control_changed(i,value/10))
            slider.sliderReleased.connect(self._apply_camera_controls)
            row.addWidget(entry); row.addWidget(slider,1); camera_layout.addLayout(row)
            self.camera_entries.append(entry); self.camera_sliders.append(slider)
        self.camera_entries[0].setToolTip('Camera position around +z: 0 [°] toward +x, 90 [°] toward +y.')
        self.camera_entries[1].setToolTip('Camera elevation above the XY plane; positive values look from above.')
        tuning_layout.addWidget(self.camera_controls); self.camera_controls.hide()
        self.splitter.addWidget(self.render_host)
        slices = QWidget(); slice_layout = QVBoxLayout(slices)
        slice_layout.setContentsMargins(4,4,4,4)
        heading = QLabel('XY Slice'); heading.setStyleSheet('font-size: 15px; font-weight: 600;')
        slice_layout.addWidget(heading)
        zrow = QHBoxLayout(); label = QLabel('Slice z')
        zrow.addWidget(label)
        self.slice_z = DecimalSpinBox(); self.slice_z.setDecimals(4)
        self.slice_z.setSuffix(' [µm]'); self.slice_z.setMinimumWidth(130)
        self.slice_z.setKeyboardTracking(False)
        self.slice_z.setToolTip('Selects the nearest calculated z plane; its actual coordinate is displayed.')
        self.slice_z.valueChanged.connect(self.select_numeric_slice)
        zrow.addWidget(self.slice_z)
        slider = QSlider(Qt.Orientation.Horizontal); slider.setRange(0,0)
        zrow.addWidget(slider,1); slice_layout.addLayout(zrow)
        self.sliders = [('z',slider,label)]
        slider.valueChanged.connect(self.schedule_slice)
        slider.sliderReleased.connect(self.draw_slices)
        slice_layout.addWidget(self.canvas, 1)
        self.splitter.addWidget(slices)
        self.splitter.setStretchFactor(0,1); self.splitter.setStretchFactor(1,1)
        self.splitter.setSizes([500,500]); layout.addWidget(self.splitter, 1)
        self.component.currentIndexChanged.connect(self.update_component)
        self.shared_scale.toggled.connect(self.update_component)
        for widget in (self.threshold, self.gamma, self.opacity):
            widget.valueChanged.connect(self.update_transfer)
        self.reset.clicked.connect(self.reset_view)
        self.show_plane.toggled.connect(self.update_overlays)
        self.plane_opacity.valueChanged.connect(self.update_overlays)
        self.show_stack.toggled.connect(self.update_overlays)
        self.stack_opacity.valueChanged.connect(self.update_overlays)
        self.splitter.setVisible(False)
        self.empty_message = QLabel(calculation_warning('XYZ volume','Compute XYZ volume'))
        self.empty_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_message.setWordWrap(True)
        self.empty_message.setStyleSheet(WARNING_STYLE)
        self.empty_host = QWidget(); empty_layout = QVBoxLayout(self.empty_host)
        empty_layout.addStretch(1); empty_layout.addWidget(self.empty_message)
        self.compute_button = QPushButton('Compute XYZ volume')
        self.compute_button.setEnabled(False)
        self.compute_button.clicked.connect(lambda:self.compute_requested.emit())
        empty_layout.addWidget(self.compute_button,0,Qt.AlignmentFlag.AlignHCenter)
        empty_layout.addStretch(1); layout.addWidget(self.empty_host,1)
        self.set_volume_available(False)
        self.vtk_error = None

    def set_volume_available(self, available):
        for widget in (self.component_controls,self.component_title,self.splitter):
            widget.setVisible(available)
        self.empty_host.setVisible(not available)

    @staticmethod
    def _setting_pair(name, widget):
        pair = QWidget(); row = QHBoxLayout(pair)
        row.setContentsMargins(0,0,0,0); row.setSpacing(4)
        row.addWidget(QLabel(name)); row.addWidget(widget)
        pair.setSizePolicy(QSizePolicy.Policy.Fixed,QSizePolicy.Policy.Preferred)
        return pair

    @staticmethod
    def _spin(value, low, high, decimals, percentage=False):
        widget = PercentageSpinBox() if percentage else DecimalSpinBox()
        widget.setDecimals(max(1,decimals-2) if percentage else decimals); widget.setRange(low, high)
        widget.setSingleStep(.01 if percentage or decimals == 3 else .1); widget.setValue(value)
        widget.setToolTip('Display only; changing this does not rerun the simulation')
        return widget

    def _ensure_renderer(self):
        if self.vtk_widget is not None or self.vtk_error is not None:
            return
        try:
            import vtkmodules.qt
            vtkmodules.qt.PyQtImpl = 'PyQt6'
            from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
            from .volume_render import VolumeScene
            self.render_frame = QFrame(self.render_host)
            self.render_frame.setPalette(self.render_host.palette())
            self.render_frame.setAutoFillBackground(True)
            frame_layout = QVBoxLayout(self.render_frame)
            frame_layout.setContentsMargins(2,2,2,2); frame_layout.setSpacing(0)
            self.vtk_widget = QVTKRenderWindowInteractor(self.render_frame)
            self.scene = VolumeScene()
            self.vtk_widget.GetRenderWindow().AddRenderer(self.scene.renderer)
            frame_layout.addWidget(self.vtk_widget)
            self.render_layout.addWidget(self.render_frame,1)
        except ImportError as exc:
            self.vtk_error = str(exc)
            self.render_layout.addWidget(QLabel('Install the GUI dependencies including vtk for the rotatable 3D view.\n'
                                               'The calculated volume and slice views remain available.'))

    def set_palette(self, name):
        self.int_color = colormaps[name.lower() if name != 'Grayscale' else 'gray']
        if self.values is not None:
            self.update_transfer(); self.draw_slices()

    def showEvent(self, event):
        super().showEvent(event)
        self.vtk_refresh_timer.start(0)

    def hideEvent(self, event):
        self.vtk_refresh_timer.stop()
        super().hideEvent(event)

    def _refresh_3d_view(self):
        """Repaint the native viewport after Qt has exposed the tab and its layout."""
        if not self.isVisible() or not self.vtk_initialized or self.values is None:
            return
        self.render_host.update()
        self.render_frame.update()
        # QVTK paintEvent performs Render; queue one full repaint instead of
        # rendering here and then rendering the same view again in paintEvent.
        self.vtk_widget.update()

    def render(self, name, result):
        if result is self.result:
            self.vtk_refresh_timer.start(0)
            return
        self.result = result
        if result.volume is None:
            self.values = None
            self.slice_timer.stop()
            self.set_volume_available(False)
            return
        self.peak = volume_peak(result.volume)
        self._ensure_renderer(); self.set_volume_available(True)
        self.slice_z.blockSignals(True)
        self.slice_z.setRange(float(result.volume_z[0]), float(result.volume_z[-1]))
        self.slice_z.setSingleStep(float(result.volume_z[1]-result.volume_z[0]))
        self.slice_z.blockSignals(False)
        g = result.config.sampling
        coords = {'x':result.volume_x, 'y':result.volume_y, 'z':result.volume_z}
        for axis, slider, _ in self.sliders:
            slider.blockSignals(True)
            slider.setRange(0, len(coords[axis])-1)
            slider.setValue(int(np.argmin(abs(coords[axis]-getattr(g, axis+'0')))))
            slider.blockSignals(False)
        self.slice_artist = None; self.slice_colorbar = None
        self.update_component()
        if self.scene is not None:
            self.scene.set_stack(result.config.stack, self.show_stack.isChecked(), self.stack_opacity.value())
        self.reset_view()

    def invalidate(self):
        self.camera_timer.stop()
        self.vtk_refresh_timer.stop()
        self.slice_timer.stop()
        self.result = None; self.values = None; self.peak = None
        self.slice_artist = None; self.slice_colorbar = None
        if self.scene is not None:
            self.scene.mapper.RemoveAllInputConnections(0)
            self.scene.outline.RemoveAllInputConnections(0)
            self.scene.image = None
            self.scene.clear_stack()

    def update_component(self, *_):
        if self.result is None or self.result.volume is None:
            return
        component = self.component.currentIndex()-1
        self.values = volume_intensity(self.result.volume, None if component < 0 else component, self.peak)
        maximum = float(self.values.max())
        self.high = 1. if self.shared_scale.isChecked() or maximum <= 0 else maximum

        if self.scene is not None:
            r = self.result

            # Limit the resolution of the 3D renderer only.
            # Keep the original scientific volume unchanged.
            source = self.values
            target_shape = (min(source.shape[0], 61), 61, 61)
            if source.shape != target_shape:
                factors = tuple(t / n for t, n in zip(target_shape, source.shape))
                render_values = zoom(source, factors, order=1, mode='nearest', grid_mode=True, prefilter=False)
            else:
                render_values = source

            # Uniform physical coordinates for the VTK image
            vx = np.linspace(
                r.volume_x[0],
                r.volume_x[-1],
                render_values.shape[2]
            )

            vy = np.linspace(
                r.volume_y[0],
                r.volume_y[-1],
                render_values.shape[1]
            )

            vz = np.linspace(
                r.volume_z[0],
                r.volume_z[-1],
                render_values.shape[0]
            )

            self.scene.set_data(
                np.ascontiguousarray(
                    render_values,
                    dtype=np.float32
                ),
                vx, vy, vz
            )

        self.component_title.setText(self.component.currentText()+' intensity')
        self.update_transfer(); self.draw_slices()

    def update_transfer(self, *_):
        if self.scene is not None and self.values is not None:
            self.scene.set_transfer(self.int_color, self.threshold.value(), self.gamma.value(),
                                    self.opacity.value(), high=self.high)
            if not self.vtk_initialized:
                self.vtk_widget.Initialize()
                from .volume_render import make_turntable_style
                self.interaction_style = make_turntable_style(self.scene)
                self.interaction_style.AddObserver('InteractionEvent',self._sync_camera_controls)
                self.interaction_style.AddObserver('EndInteractionEvent',self._sync_camera_controls)
                self.vtk_widget.GetRenderWindow().GetInteractor().SetInteractorStyle(self.interaction_style)
                self.vtk_initialized = True
            self.vtk_widget.GetRenderWindow().Render()

    def _toggle_camera_controls(self, expanded):
        self.camera_controls.setVisible(expanded)
        self.camera_toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        if expanded: self._sync_camera_controls()

    def _camera_control_changed(self, index, value):
        entry,slider = self.camera_entries[index],self.camera_sliders[index]
        entry.blockSignals(True); entry.setValue(value); entry.blockSignals(False)
        slider.blockSignals(True); slider.setValue(round(value*10)); slider.blockSignals(False)
        if self.values is not None and not self.camera_timer.isActive(): self.camera_timer.start()

    def _apply_camera_controls(self):
        self.camera_timer.stop()
        if self.scene is None or self.values is None: return
        self.scene.set_camera_angles(*(entry.value() for entry in self.camera_entries))
        self._sync_camera_controls()
        self.vtk_widget.GetRenderWindow().Render()

    def _sync_camera_controls(self, *_):
        if self.scene is None or self.values is None: return
        # During a slider edit, retain the latest requested angle until its timer applies it.
        if self.camera_timer.isActive(): return
        angles = self.scene.camera_angles()
        elevation = float(np.clip(angles[1],-89,89))
        if elevation != angles[1]:
            self.scene.set_camera_angles(angles[0],elevation)
            angles = (angles[0],elevation)
        for entry,slider,value in zip(self.camera_entries,self.camera_sliders,angles):
            entry.blockSignals(True); entry.setValue(value); entry.blockSignals(False)
            slider.blockSignals(True); slider.setValue(round(entry.value()*10)); slider.blockSignals(False)

    def reset_view(self):
        if self.scene is not None and self.values is not None:
            self.camera_timer.stop()
            self.scene.reset_camera(); self._sync_camera_controls()
            self.vtk_widget.GetRenderWindow().Render()

    def select_numeric_slice(self, value):
        if self.result is None or self.values is None:
            return
        index = int(np.argmin(np.abs(self.result.volume_z-value)))
        slider = self.sliders[0][1]
        slider.blockSignals(True); slider.setValue(index); slider.blockSignals(False)
        self.draw_slices()

    def schedule_slice(self, *_):
        """Coalesce drag events, always drawing the newest slider position."""
        if self.values is None:
            return
        self.sync_slice_entry()
        if not self.slice_timer.isActive():
            self.slice_timer.start()

    def sync_slice_entry(self):
        iz = self.sliders[0][1].value()
        self.slice_z.blockSignals(True)
        self.slice_z.setValue(float(self.result.volume_z[iz]))
        self.slice_z.blockSignals(False)

    def update_overlays(self, *_):
        if self.scene is None or self.values is None:
            return
        iz = self.sliders[0][1].value()
        self.scene.set_slice_plane(float(self.result.volume_z[iz]), self.show_plane.isChecked(),
                                   self.plane_opacity.value())
        self.scene.set_stack_visible(self.show_stack.isChecked())
        self.scene.set_stack_opacity(self.stack_opacity.value())
        if self.vtk_initialized:
            self.vtk_widget.GetRenderWindow().Render()

    def draw_slices(self, *_):
        if self.values is None:
            return
        self.slice_timer.stop()
        r = self.result; iz = self.sliders[0][1].value()
        self.sync_slice_entry()
        # Reuse the axes, image and colorbar instead of rebuilding the figure on every drag.
        if self.slice_artist is None:
            self.figure.clear(); ax = self.figure.subplots()
            self.slice_artist = ax.imshow(self.values[iz], origin='lower', aspect='equal',
                vmin=0, vmax=self.high, cmap=self.int_color,
                extent=(r.volume_x[0],r.volume_x[-1],r.volume_y[0],r.volume_y[-1]))
            ax.set(xlabel='x [µm]', ylabel='y [µm]')
            cax = ax.inset_axes((1.04,0,.045,1), transform=ax.transAxes)
            self.slice_colorbar = self.figure.colorbar(self.slice_artist,cax=cax,label='intensity')
            self.slice_colorbar.ax.tick_params(labelsize=9)
        else:
            self.slice_artist.set_data(self.values[iz])
            self.slice_artist.set_extent((r.volume_x[0],r.volume_x[-1],r.volume_y[0],r.volume_y[-1]))
            self.slice_artist.set_cmap(self.int_color)
            self.slice_artist.set_clim(0,self.high)
        self.canvas.draw_idle()
        self.update_overlays()

    def export_session(self, folder, result, fmt, dpi, progress=None):
        self.render('7 · XYZ volume', result)
        if result.volume is None:
            return None
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        export_volume_data(folder, result, progress)
        selected = self.component.currentIndex()
        from PIL import Image
        images = []
        try:
            for index, stem in enumerate(('total_intensity','ex_intensity','ey_intensity','ez_intensity')):
                self.component.blockSignals(True)
                self.component.setCurrentIndex(index)
                self.component.blockSignals(False)
                self.update_component()
                iz = self.sliders[0][1].value()
                slices = ((self.values[iz], result.volume_x, result.volume_y, 'x', 'y', 'xy', result.volume_z[iz]),)
                for values, horizontal, vertical, hn, vn, plane, fixed in slices:
                    figure = Figure(figsize=(6.5,5), layout='constrained'); ax = figure.subplots()
                    artist = ax.imshow(values, origin='lower', aspect='equal', cmap=self.int_color,
                                       vmin=0, vmax=self.high, extent=(horizontal[0],horizontal[-1],vertical[0],vertical[-1]))
                    ax.set(title=f'{self.component.currentText()} intensity · {plane.upper()} slice',
                           xlabel=f'{hn} [µm]', ylabel=f'{vn} [µm]')
                    figure.colorbar(artist, ax=ax, label='intensity')
                    figure.savefig(folder/f'{stem}_{plane}.{fmt}', dpi=dpi)
                    hh,vv = np.meshgrid(horizontal,vertical)
                    np.savetxt(folder/f'{stem}_{plane}.txt',
                               np.column_stack((hh.ravel(),vv.ravel(),values.ravel())),
                               header=f'{plane.upper()} slice; fixed coordinate = {fixed:.17g} um\n'
                                      f'columns: {hn}_um {vn}_um display_intensity', fmt='%.17g')
                if self.scene is not None:
                    from .volume_render import capture_image
                    image = capture_image(self.vtk_widget.GetRenderWindow())
                    image.save(folder/f'{stem}_volume.{fmt}', dpi=(dpi,dpi))
                    thumb = image.copy(); thumb.thumbnail((600, 450)); images.append(thumb)
        finally:
            self.component.blockSignals(True); self.component.setCurrentIndex(selected)
            self.component.blockSignals(False); self.update_component()
        import json
        camera = self.scene.renderer.GetActiveCamera() if self.scene is not None else None
        (folder/'display_settings.json').write_text(json.dumps({
            'threshold':self.threshold.value(), 'opacity_exponent':self.gamma.value(),
            'max_opacity':self.opacity.value(), 'colormap':self.int_color.name,
            'common_total_peak':self.peak,
            'xy_slice_index_z':self.sliders[0][1].value(),
            'xy_slice_z_um':float(result.volume_z[self.sliders[0][1].value()]),
            'show_xy_plane':self.show_plane.isChecked(),
            'slice_plane_opacity':self.plane_opacity.value(),
            'show_layer_stack':self.show_stack.isChecked(),
            'layer_stack_opacity':self.stack_opacity.value(),
            'shared_component_scale':self.shared_scale.isChecked(),
            'selected_color_scale_max':self.high,
            'rendering':'CPU ray casting' if self.scene is not None else 'XY slice only',
            'component':self.component.currentText(),
            'camera': {'position':camera.GetPosition(), 'focal_point':camera.GetFocalPoint(),
                       'view_up':camera.GetViewUp(),
                       'azimuth_deg':self.scene.camera_angles()[0],
                       'elevation_deg':self.scene.camera_angles()[1],
                       'interaction':'turntable'} if camera is not None else None,
        }, indent=2), encoding='utf-8')
        overview = BytesIO()
        if images:
            montage = Image.new('RGB', (1200,900), 'white')
            for j,image in enumerate(images): montage.paste(image, ((j%2)*600,(j//2)*450))
            montage.save(overview, format='PNG')
        else:
            self.figure.savefig(overview, format='png', dpi=150)
        overview.seek(0)
        return overview

    def shutdown(self):
        self.camera_timer.stop()
        self.vtk_refresh_timer.stop()
        self.slice_timer.stop()
        if self.vtk_widget is not None:
            self.vtk_widget.Finalize()
