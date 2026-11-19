"""Modern Qt layout for the existing VTK scientific volume renderer.

Only the presentation is changed. The VTK renderer, transfer functions,
scientific data and volume-calculation code are inherited unmodified from
stratified_focus.volume_gui.VolumePanel.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QGridLayout, QLabel, QGroupBox,
    QSizePolicy, QScrollArea, QFrame,
)

from stratified_focus.volume_gui import VolumePanel
from .widgets import MapView


class SquareSliceFrame(QWidget):
    """Center a square XY plot without imposing a minimum on the window."""

    def __init__(self, plot: MapView, parent=None):
        super().__init__(parent)
        self.plot = plot
        self.plot.setParent(self)
        self.setMinimumSize(0, 0)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.plot.show()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep outer plot square and comfortably contained in its panel.
        side = max(1, min(self.width() - 8, self.height() - 8, 440))
        self.plot.setGeometry(
            (self.width() - side) // 2,
            (self.height() - side) // 2,
            side,
            side,
        )


class ModernVolumePanel(VolumePanel):
    """VTK volume on the left; compact controls and a PyQtGraph XY slice on the right."""

    def __init__(self):
        super().__init__()
        self._first_show = True

        # Reuse the original VTK host as-is: the native VTK widget is still
        # created by VolumePanel._ensure_renderer() inside this same host.
        left_layout = self.render_layout
        heading_item = left_layout.itemAt(0)
        if heading_item is not None and heading_item.widget() is not None:
            heading = heading_item.widget()
            left_layout.removeWidget(heading)
            heading.hide()
            self._old_volume_heading = heading

        original_settings = self.render_host.findChild(QGroupBox)
        if original_settings is None:
            raise RuntimeError('Original VTK visualization settings were not found')
        left_layout.removeWidget(original_settings)
        original_settings.hide()
        # Keep the old Qt setting-pair containers alive. The existing controls
        # are transferred below without disconnecting their signals.
        original_settings.setParent(self)
        self._old_settings = original_settings
        left_layout.setContentsMargins(2, 2, 2, 2)
        left_layout.setSpacing(0)

        right = self.splitter.widget(1)
        right_layout = right.layout()
        right_layout.setContentsMargins(6, 4, 6, 4)
        right_layout.setSpacing(5)

        # The original Matplotlib canvas stays available for legacy exports,
        # but the on-screen XY slice uses exactly the same MapView as the tabs.
        right_layout.removeWidget(self.canvas)
        self.canvas.setParent(self)
        self.canvas.hide()

        # Move the selectors and the existing live title to the right panel.
        root = self.layout()
        root.removeWidget(self.component_controls)
        root.removeWidget(self.component_title)
        self.component_controls.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.shared_scale.setText('Shared Scale')
        self.component_title.setStyleSheet('font-size: 11pt; font-weight: 600;')

        # Build a compact settings box while reusing every existing control and
        # every original signal connection (including camera interactions).
        settings = QGroupBox('Visualization Settings')
        grid = QGridLayout(settings)
        grid.setContentsMargins(8, 12, 8, 8)
        grid.setHorizontalSpacing(7)
        grid.setVerticalSpacing(5)

        def add_spin(row, label, widget):
            widget.setMinimumWidth(75)
            widget.setMaximumWidth(115)
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(widget, row, 1)

        add_spin(0, 'Threshold', self.threshold)
        add_spin(1, 'Gamma', self.gamma)
        add_spin(2, 'Opacity', self.opacity)
        def opacity_pair(widget):
            host = QWidget()
            layout = QHBoxLayout(host)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)
            layout.addWidget(QLabel('Opacity'))
            layout.addWidget(widget, 1)
            widget.setMinimumWidth(70)
            widget.setMaximumWidth(100)
            return host

        grid.addWidget(self.show_plane, 3, 0)
        grid.addWidget(opacity_pair(self.plane_opacity), 3, 1)
        self.plane_opacity.setToolTip('XY slice plane opacity')
        grid.addWidget(self.show_stack, 4, 0)
        grid.addWidget(opacity_pair(self.stack_opacity), 4, 1)
        self.stack_opacity.setToolTip('Layer interface opacity')
        grid.addWidget(self.camera_toggle, 5, 0, 1, 2)
        grid.addWidget(self.camera_controls, 6, 0, 1, 2)
        self.camera_controls.hide()
        for entry in self.camera_entries:
            entry.setMinimumWidth(65)
            entry.setMaximumWidth(88)
        grid.addWidget(self.reset, 7, 0, 1, 2)
        grid.setColumnStretch(0, 1)

        # The original slice heading and z-slider row are already in the
        # correct place and retain their original connections.
        right_layout.insertWidget(0, self.component_controls)
        right_layout.insertWidget(1, settings)
        right_layout.insertWidget(2, self.component_title)
        self.component_controls.show()

        self.slice_map = MapView('x [µm]', 'y [µm]')
        self.slice_map.plot.setAspectLocked(True)


        self.slice_frame = SquareSliceFrame(self.slice_map)

        # Reserve enough vertical space for the complete XY plot,
        # including axes, title and colorbar.
        self.slice_frame.setFixedHeight(400)
        self.slice_frame.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed
        )

        right_layout.addWidget(self.slice_frame)

        # Make the complete right panel vertically scrollable.
        # All existing widgets and signal connections are preserved.
        self.right_scroll = QScrollArea()
        self.right_scroll.setFrameShape(
            QFrame.Shape.NoFrame
        )
        self.right_scroll.setWidgetResizable(True)

        self.right_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.right_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self.right_scroll.setMinimumWidth(360)

        # Replace only the right splitter widget.
        # The original VTK renderer remains untouched.
        old_right = self.splitter.replaceWidget(
            1, self.right_scroll
        )
        self.right_scroll.setWidget(old_right)


        # The right panel may shrink below its size hint; this is a draggable
        # 68/32 split, not a fixed pixel split.
        self.splitter.setHandleWidth(6)
        self.splitter.setStretchFactor(0, 68)
        self.splitter.setStretchFactor(1, 32)
        right.setMinimumWidth(280)
        self.render_host.setMinimumWidth(300)

    def showEvent(self, event):
        super().showEvent(event)
        if self._first_show:
            self._first_show = False
            QTimer.singleShot(0, self._set_initial_split)

    def _set_initial_split(self):
        width = self.splitter.width()
        if width > 0:
            self.splitter.setSizes([round(width * 0.68), round(width * 0.32)])

    def draw_slices(self, *_):
        """Display XY values in the same PyQtGraph widget used by all tabs."""
        if self.values is None or self.result is None:
            return
        # Preserve the original offscreen Matplotlib canvas for legacy exports.
        # The visible XY plot below is always PyQtGraph.
        super().draw_slices(*_)
        r = self.result
        iz = self.sliders[0][1].value()
        self.sync_slice_entry()

        self.slice_map.set_data(
            self.values[iz],
            r.volume_x,
            r.volume_y,
            cmap=self.int_color.name,
            levels=(0.0, max(float(self.high), 1e-12)),
            label='Normalized intensity',
        )
        self.slice_map.set_title(
            f'{self.component.currentText()} intensity · '
            f'z = {float(r.volume_z[iz]):+.3f} [µm]'
        )

    def apply_theme(self, name):
        self.slice_map.apply_theme(name)
