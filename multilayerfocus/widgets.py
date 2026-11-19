"""Reusable Qt widgets: cards, sliders, maps, stack diagram, metric tiles."""
import math
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QPointF, QSize, QTimer
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush, QFont, QLinearGradient, QPainterPath, QPolygonF
from PyQt6.QtWidgets import (QFrame, QVBoxLayout, QHBoxLayout, QLabel, QToolButton, QWidget, QSlider,
                             QDoubleSpinBox, QSizePolicy, QSpinBox)
from .theme import PALETTES
from .plot_interaction import UnifiedViewBox, setup_unified_plot
from html import escape
pg.setConfigOptions(imageAxisOrder='row-major', antialias=True)


class Card(QFrame):
    """Collapsible titled panel."""
    def __init__(self, title, subtitle='', expanded=True):
        super().__init__()
        self.setObjectName('card')
        lay = QVBoxLayout(self); lay.setContentsMargins(7, 6, 7, 6); lay.setSpacing(5)
        head = QHBoxLayout(); head.setSpacing(6)
        self.toggle = QToolButton(); self.toggle.setObjectName('cardHeader')
        self.toggle.setText(title)
        self.toggle.setCheckable(True); self.toggle.setChecked(expanded)
        self.toggle.toggled.connect(self._toggled)
        self.toggle.setStyleSheet('font-weight: 600; font-size: 10pt;')
        # Larger, clickable expand/collapse arrow
        self.arrow = QToolButton()
        self.arrow.setObjectName('cardArrow')
        self.arrow.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.arrow.setFixedWidth(28)
        self.arrow.clicked.connect(self.toggle.click)
        head.addWidget(self.arrow)
        head.addWidget(self.toggle)
        head.addStretch(1)

        self.extra = QHBoxLayout(); head.addLayout(self.extra)
        lay.addLayout(head)
        self.subtitle = QLabel(subtitle); self.subtitle.setObjectName('cardSubtitle'); self.subtitle.setWordWrap(True)
        if subtitle: lay.addWidget(self.subtitle)
        self.body = QWidget(); self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(0, 2, 0, 0); self.body_layout.setSpacing(6)
        lay.addWidget(self.body)
        self.body.setVisible(expanded)
        self._title = title

    def _toggled(self, on):
        self.body.setVisible(on)
        self.toggle.setText(self._title)
        self.arrow.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)

    def add(self, w):
        self.body_layout.addWidget(w); return w

    def add_layout(self, l):
        self.body_layout.addLayout(l); return l


class Spin(QDoubleSpinBox):
    def __init__(self, value, lo, hi, decimals=3, step=None, suffix=''):
        super().__init__()
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)        
        self.setRange(lo, hi); self.setDecimals(decimals); self.setValue(value)
        self.setKeyboardTracking(False)
        self.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.UpDownArrows)
        self.setStyleSheet("""
        QDoubleSpinBox::up-button,
        QDoubleSpinBox::down-button {
        width: 0px;
        border: none;
          }
        """)
        if step: self.setSingleStep(step)
        if suffix:
            unit = suffix.strip()
            self.setSuffix(' ' + (unit if unit.startswith('[') else f'[{unit}]'))
        self.setAlignment(Qt.AlignmentFlag.AlignRight)

    def wheelEvent(self, e):          # avoid accidental changes while scrolling the side panel
        if self.hasFocus(): super().wheelEvent(e)
        else: e.ignore()


class ISpin(QSpinBox):
    def __init__(self, value, lo, hi, suffix=''):
        super().__init__()
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)        
        self.setRange(lo, hi); self.setValue(value); self.setKeyboardTracking(False)
        self.setButtonSymbols(QSpinBox.ButtonSymbols.UpDownArrows)
        self.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.setStyleSheet("""
        QSpinBox::up-button,
        QSpinBox::down-button {
        width: 0px;
        border: none;
        }
        """)

        if suffix:
            unit = suffix.strip()
            self.setSuffix(' ' + (unit if unit.startswith('[') else f'[{unit}]'))

    def wheelEvent(self, e):
        if self.hasFocus(): super().wheelEvent(e)
        else: e.ignore()


class SliderSpin(QWidget):
    """Label + slider + numeric field kept in sync; emits valueChanged(float)."""
    valueChanged = pyqtSignal(float)

    def __init__(self, label, value, lo, hi, decimals=3, unit='', log=False,
                 step=None, tooltip='', compact=False):
        super().__init__()
        self.lo, self.hi, self.log = lo, hi, log
        self.label = QLabel(label)
        self.spin = Spin(value, lo, hi, decimals, step, (' ' + unit) if unit else '')
        self.spin.setFixedWidth(104)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)

        if compact:
            # One-line slider for the Full volume control strip only.
            lay = QHBoxLayout(self)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(7)
            if label:
                lay.addWidget(self.label)
            lay.addWidget(self.slider, 1)
            lay.addWidget(self.spin)
        else:
            # Preserve the approved sidebar widget geometry unchanged.
            lay = QVBoxLayout(self)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(2)
            top = QHBoxLayout()
            top.setSpacing(6)
            top.addWidget(self.label)
            top.addStretch(1)
            top.addWidget(self.spin)
            lay.addLayout(top)
            lay.addWidget(self.slider)
        self.slider.setValue(self._to_slider(value))
        self.slider.valueChanged.connect(self._slider_moved)
        self.spin.valueChanged.connect(self._spin_changed)
        if tooltip:
            self.setToolTip(tooltip)

    def _to_slider(self, v):
        if self.log:
            v = min(max(v, self.lo), self.hi)
            return int(round(1000*math.log(v/self.lo)/math.log(self.hi/self.lo)))
        return int(round(1000*(v-self.lo)/(self.hi-self.lo)))

    def _from_slider(self, s):
        if self.log:
            return self.lo*(self.hi/self.lo)**(s/1000)
        return self.lo + (self.hi-self.lo)*s/1000

    def _slider_moved(self, s):
        v = self._from_slider(s)
        self.spin.blockSignals(True); self.spin.setValue(v); self.spin.blockSignals(False)
        self.valueChanged.emit(self.spin.value())

    def _spin_changed(self, v):
        self.slider.blockSignals(True); self.slider.setValue(self._to_slider(v)); self.slider.blockSignals(False)
        self.valueChanged.emit(v)

    def value(self):
        return self.spin.value()

    def setValue(self, v):
        self.spin.blockSignals(True); self.slider.blockSignals(True)
        self.spin.setValue(v); self.slider.setValue(self._to_slider(v))
        self.spin.blockSignals(False); self.slider.blockSignals(False)

    def setRange(self, lo, hi):
        v = self.value(); self.lo, self.hi = lo, hi; self.spin.setRange(lo, hi); self.setValue(min(max(v, lo), hi))


def row(label, widget, tooltip=''):
    w = QWidget(); l = QHBoxLayout(w); l.setContentsMargins(0, 0, 0, 0); l.setSpacing(5)
    lab = QLabel(label)
    lab.setWordWrap(True)
    lab.setMinimumWidth(0)
    lab.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    l.addWidget(lab, 1)
    l.addWidget(widget)
    if tooltip: w.setToolTip(tooltip)
    return w


class MetricTile(QFrame):
    def __init__(self, caption, value='—', tooltip=''):
        super().__init__(); self.setObjectName('tile')
        l = QVBoxLayout(self); l.setContentsMargins(8, 6, 8, 6); l.setSpacing(0)
        self.value = QLabel(value); self.value.setObjectName('tileValue')
        self.caption = QLabel(caption); self.caption.setObjectName('tileCaption')
        l.addWidget(self.value); l.addWidget(self.caption)
        if tooltip: self.setToolTip(tooltip)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set(self, text, color=None):
        self.value.setText(text)
        self.value.setStyleSheet(f'color: {color};' if color else '')


CMAPS = ['inferno', 'magma', 'viridis', 'cividis', 'turbo', 'plasma', 'gray']


THEME = {'name': 'light'}


def get_cmap(name):
    if name == 'phase':
        return pg.colormap.get('twilight', source='matplotlib')
    if name == 'diverging':
        return pg.colormap.get('berlin' if THEME['name'] == 'dark' else 'RdBu_r', source='matplotlib')
    return pg.colormap.get(name, source='matplotlib')


class MapView(pg.GraphicsLayoutWidget):
    """2D map with color bar, interface lines, plane marker, click and hover signals."""
    clicked = pyqtSignal(float, float)
    hovered = pyqtSignal(float, float, float)

    def __init__(self, xlabel='x [µm]', ylabel='y [µm]', theme='light'):
        super().__init__()
        # Never let a stale plot size hint impose a minimum on the tab.
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.theme = theme
        self.plot = self.addPlot(viewBox=UnifiedViewBox())
        setup_unified_plot(self.plot, owner=self)
        self.plot.setLabel('bottom', xlabel); self.plot.setLabel('left', ylabel)
        self.plot.showAxis('bottom', True)
        self.plot.showAxis('left', True)
        # Reserve a small outside margin for complete axis captions.
        self.ci.layout.setContentsMargins(2, 2, 4, 4)
        self.plot.setAspectLocked(True)
        # Identical axis geometry for all maps prevents the square data area
        # from changing size when a colorbar uses different tick labels.
        self.plot.getAxis('left').setWidth(47)
        self.plot.getAxis('bottom').setHeight(41)
        self.plot.hideButtons()
        self.img = pg.ImageItem()
        self.plot.addItem(self.img)
        self.bar = pg.ColorBarItem(values=(0, 1), colorMap=get_cmap('inferno'), width=12, interactive=False)
        self.bar.setImageItem(self.img, insert_in=self.plot)
        # Equal tick-label columns also prevent scientific notation from
        # running into the neighboring panel (especially in Energy).
        self.bar.axis.setWidth(50)
        self.lines = []; self.labels = []
        self.marker = None; self.vmarker = None
        self.title = pg.LabelItem('', justify='left')
        self.x = self.y = self.values = None
        self._reframe_pending = False
        self.scene().sigMouseClicked.connect(self._click)
        self.scene().sigMouseMoved.connect(self._move)
        self.apply_theme(theme)

    def apply_theme(self, theme):
        self.theme = theme; p = PALETTES[theme]; THEME['name'] = theme
        self.setBackground(p['plot_bg'])
        for ax in ('left', 'bottom'):
            a = self.plot.getAxis(ax); a.setPen(pg.mkPen(p['plot_fg'])); a.setTextPen(pg.mkPen(p['plot_fg']))
            a.enableAutoSIPrefix(False)
            a.setTickFont(QFont('Segoe UI', 8))
        a = self.bar.axis; a.setPen(pg.mkPen(p['plot_fg'])); a.setTextPen(pg.mkPen(p['plot_fg']))
        a.enableAutoSIPrefix(False)
        a.setTickFont(QFont('Segoe UI', 8))
        # Keep text visually close to the colorbar, leaving enough room for
        # full numeric tick labels and their physical units.
        a.setStyle(tickTextOffset=3)

    def set_title(self, text):
        self.export_title = str(text)
        self.plot.setTitle(
            f'<span style="font-size:9pt; font-weight:700;">'
            f'{escape(str(text))}</span>',
            color=PALETTES[self.theme]['plot_fg']
        )

    def set_data(self, values, x, y, cmap='inferno', levels=None, log=False, label=''):
        v = np.asarray(values, float)
        self.values, self.x, self.y = v, np.asarray(x), np.asarray(y)
        self.display_unit = str(label)
        self.display_log = bool(log)
        disp = v
        if log:
            pos = v[v > 0]
            floor = (pos.max()*1e-4) if pos.size else 1e-30
            disp = np.log10(np.maximum(v, floor))
            levels = (math.log10(floor), math.log10(pos.max() if pos.size else 1))
        if levels is None:
            lo, hi = float(np.nanmin(disp)), float(np.nanmax(disp))
            if hi <= lo: hi = lo + 1e-30
            levels = (lo, hi)
        self.img.setImage(disp, autoLevels=False)
        self.img.setRect(QRectF(float(x[0]), float(min(y[0], y[-1])), float(x[-1]-x[0]), float(abs(y[-1]-y[0]))))
        self.bar.setColorMap(get_cmap(cmap))
        self.bar.setLevels(levels)
        self.bar.axis.setLabel(('log₁₀ ' if log else '') + label)
        self.plot.setXRange(float(x[0]), float(x[-1]), padding=0)
        self.plot.setYRange(float(min(y[0], y[-1])), float(max(y[0], y[-1])), padding=0)
        self.schedule_reframe()

        # Do not freeze GraphicsLayout rows or columns in pixels.
        # The viewport and colorbar must follow the parent frame on resize.

    def schedule_reframe(self):
        """Reframe after Qt has finalized the plot's onscreen geometry."""
        if self._reframe_pending:
            return
        self._reframe_pending = True
        QTimer.singleShot(0, self.reframe_data)

    def reframe_data(self):
        self._reframe_pending = False
        if self.x is None or self.y is None or not len(self.x) or not len(self.y):
            return
        # Use actual image extent, not overlays, annotations or a prior zoom.
        x0, x1 = float(self.x[0]), float(self.x[-1])
        y0, y1 = float(min(self.y[0], self.y[-1])), float(max(self.y[0], self.y[-1]))
        if x0 == x1 or y0 == y1:
            return
        vb = self.plot.getViewBox()
        vb.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)
        vb.forget_home()
        vb.remember_home()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if getattr(self, '_reframe_pending', None) is not None and self.x is not None:
            self.schedule_reframe()

    def set_interfaces(self, items, line_color=None, label_color=None):
        """Dashed lines at interfaces; labels of closely spaced interfaces are merged."""
        for it in self.lines + self.labels:
            self.plot.removeItem(it)
        self.lines, self.labels = [], []
        p = PALETTES[self.theme]
        span = (float(np.ptp(self.y)) if self.y is not None else 1.0) or 1.0
        merged = []
        for z, label in items:
            if merged and abs(z - merged[-1][0]) < 0.06*span:
                z0, lab0, zs = merged[-1]
                merged[-1] = (z0, lab0 + ' | ' + label.split(' | ')[-1], zs + [z])
            else:
                merged.append((z, label, [z]))
        for _, _, zs in merged:
            for z in zs:
                ln = pg.InfiniteLine(
                    pos=z,
                    angle=0,
                    pen=pg.mkPen(
                        line_color if line_color is not None else p['interface'],
                        width=1.2,
                        style=Qt.PenStyle.DashLine
                    )
                )
                ln.setOpacity(1.0 if line_color is not None else 0.75)
                self.plot.addItem(ln)
                self.lines.append(ln)              
        for k, (z, label, zs) in enumerate(merged):
            if not label:
                continue
            right = k % 2 == 1
            t = pg.TextItem(
                label,
                color=label_color if label_color is not None else (255, 255, 255, 210),
                anchor=(1, 1) if right else (0, 1)
            )             
            xpos = (self.x[-1] if right else self.x[0]) if self.x is not None else 0
            t.setPos(float(xpos), z); t.setZValue(10)
            f = QFont(); f.setPointSize(8); t.setFont(f)
            self.plot.addItem(t); self.labels.append(t)

    def set_marker(self, z, color=None):
        p = PALETTES[self.theme]
        if self.marker is None:
            self.marker = pg.InfiniteLine(pos=z, angle=0, pen=pg.mkPen(color or p['accent'], width=2))
            self.plot.addItem(self.marker)
        self.marker.setValue(z)

    def _click(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton: return
        pos = self.plot.vb.mapSceneToView(ev.scenePos())
        if self.x is not None and self.x[0] <= pos.x() <= self.x[-1]:
            self.clicked.emit(float(pos.x()), float(pos.y()))

    def _move(self, scene_pos):
        if self.values is None: return
        pos = self.plot.vb.mapSceneToView(scene_pos)
        x, y = pos.x(), pos.y()
        if not (self.x[0] <= x <= self.x[-1] and min(self.y) <= y <= max(self.y)): return
        ix = int(np.argmin(abs(self.x-x))); iy = int(np.argmin(abs(self.y-y)))
        self.hovered.emit(float(x), float(y), float(self.values[iy, ix]))


class LinePlot(pg.PlotWidget):

    def setTitle(self, text, **kwargs):
        title = (
            f'<b>{escape(str(text))}</b>'
            if text is not None else None
        )
        self.getPlotItem().setTitle(title, **kwargs)

    def __init__(self, xlabel='', ylabel='', theme='light'):
        super().__init__(viewBox=UnifiedViewBox())
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setLabel('bottom', xlabel); self.setLabel('left', ylabel)
        self.showGrid(x=True, y=True, alpha=0.15)
        self.legend = self.addLegend(offset=(-10, 10))
        self.apply_theme(theme)
        self.hideButtons()
        setup_unified_plot(self.getPlotItem(), owner=self, grid=True)

    def clear(self):
        super().clear()
        self.getViewBox().forget_home()

    def apply_theme(self, theme):
        p = PALETTES[theme]; self.setBackground(p['plot_bg'])
        for ax in ('left', 'bottom'):
            a = self.getAxis(ax); a.setPen(pg.mkPen(p['plot_fg'])); a.setTextPen(pg.mkPen(p['plot_fg']))
            a.enableAutoSIPrefix(False)
        if self.legend is not None:
            self.legend.setLabelTextColor(p['plot_fg'])


def material_colour(n):
    n = complex(n)
    if n.imag > 1.0:            # metal
        return QColor(176, 184, 196)
    if n.imag > 1e-3:           # absorbing dielectric
        return QColor(240, 170, 90)
    t = min(max((n.real-1.0)/1.0, 0), 1)
    return QColor(int(220-60*t), int(232-40*t), int(250-20*t))


class StackDiagram(QWidget):
    """Schematic multilayer with focusing cone, nominal focus and the XY evaluation plane."""
    def __init__(self, theme='light'):
        super().__init__()
        self.theme = theme
        self.layers = []; self.z_out = 0.0; self.z0 = 0.0; self.na = 1.0; self.n1 = 1.0
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def sizeHint(self):
        return QSize(300, 260)

    def set_state(self, layers, z_out, z0, na, n1):
        self.layers, self.z_out, self.z0, self.na, self.n1 = layers, z_out, z0, na, n1
        self.update()

    def paintEvent(self, e):
        p = PALETTES[self.theme]
        qp = QPainter(self); qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        qp.fillRect(0, 0, W, H, QColor(p['surface']))
        if not self.layers:
            return
        N = len(self.layers)
        finite = [l.d for l in self.layers[1:-1]] if N > 2 else []
        # vertical layout: semi-infinite bands fixed height, finite layers ~ log-scaled thickness
        top_margin, bottom_margin = 14, 34
        avail = H - top_margin - bottom_margin
        if N == 1:
            heights = [avail]
        else:
            band = 0.30*avail if N > 2 else 0.5*avail
            rest = avail - 2*band if N > 2 else 0
            weights = [math.log10(1 + 50*max(d, 1e-4)) + 0.25 for d in finite]
            tot = sum(weights) or 1
            heights = [band] + [rest*w/tot for w in weights] + [band] if N > 2 else [band, band]
        # draw from bottom (incident) to top (exit)
        y = H - bottom_margin
        rects = []
        x0, x1 = 70, W - 12
        f = QFont(); f.setPointSizeF(8.0); qp.setFont(f)
        for l, h in zip(self.layers, heights):
            r = QRectF(x0, y-h, x1-x0, h)
            qp.fillRect(r, material_colour(l.n))
            rects.append(r); y -= h
        qp.setPen(QPen(QColor(p['border']), 1)); qp.setBrush(Qt.BrushStyle.NoBrush)
        for r in rects: qp.drawRect(r)
        # interface labels (z coordinates)
        st_z = self._interfaces()
        qp.setPen(QColor(p['muted']))
        for k, r in enumerate(rects[:-1]):
            zt = st_z[k] if k < len(st_z) else None
            if zt is not None:
                qp.drawText(QRectF(0, r.top()-8, x0-6, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f'{zt:g} [µm]')
        # focusing cone (schematic) converging to the nominal focus mapped into the diagram
        fy = self._map_z(0.0, rects, st_z)
        cx = (x0+x1)/2
        half = min((x1-x0)/2-6, (H-bottom_margin-fy)*math.tan(math.asin(min(self.na/self.n1, 0.999))))
        grad = QLinearGradient(0, H-bottom_margin, 0, fy)
        c1 = QColor(p['accent2']); c1.setAlpha(25); c2 = QColor(p['accent2']); c2.setAlpha(110)
        grad.setColorAt(0, c1); grad.setColorAt(1, c2)
        poly = QPolygonF([QPointF(cx-half, H-bottom_margin), QPointF(cx+half, H-bottom_margin), QPointF(cx, fy)])
        qp.setPen(Qt.PenStyle.NoPen); qp.setBrush(QBrush(grad)); qp.drawPolygon(poly)
        qp.setBrush(QColor(p['accent2'])); qp.drawEllipse(QPointF(cx, fy), 3.5, 3.5)
        # XY plane marker
        py = self._map_z(self.z0, rects, st_z)
        qp.setPen(QPen(QColor(p['accent']), 2, Qt.PenStyle.DashLine)); qp.drawLine(QPointF(x0, py), QPointF(x1, py))
        qp.setPen(QColor(p['accent'])); qp.drawText(QRectF(x0+4, py-16, 140, 14), Qt.AlignmentFlag.AlignLeft, f'XY plane z₀ = {self.z0:g} [µm]')
        # layer text
        qp.setPen(QColor('#1b2333'))
        for k, (l, r) in enumerate(zip(self.layers, rects)):
            label = l.name.split(' (')[0]
            nn = f'n = {l.n_re:g}' + (f' + {l.n_im:g}i' if l.n_im else '')
            dd = '' if l.d is None else f'   d = {l.d:g} [µm]'
            if r.height() >= 26:
                qp.drawText(r.adjusted(8, 2, -8, -2), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f'{label}\n{nn}{dd}')
            elif r.height() >= 11:
                qp.drawText(r.adjusted(8, 0, -8, 0), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f'{label}  {nn}{dd}')
        # objective
        qp.setPen(Qt.PenStyle.NoPen); qp.setBrush(QColor('#3b4352'))
        qp.drawRoundedRect(QRectF(cx-half-10, H-bottom_margin+4, 2*half+20, 18), 5, 5)
        qp.setPen(QColor('white')); qp.drawText(QRectF(cx-half-10, H-bottom_margin+4, 2*half+20, 18),
                                                Qt.AlignmentFlag.AlignCenter, f'objective  NA {self.na:g}')
        qp.end()

    def _interfaces(self):
        if len(self.layers) < 2: return []
        d = [l.d for l in self.layers[1:-1]]
        z_in = self.z_out - sum(d)
        return list(np.r_[z_in, z_in + np.cumsum(d)]) if d else [self.z_out]

    def _map_z(self, z, rects, zi):
        """Map physical z to a y pixel in the schematic (piecewise linear, compressed outside)."""
        if len(rects) == 1:
            r = rects[0]; return r.center().y() - (z/(abs(z)+1))*r.height()*0.4
        if z <= zi[0]:
            r = rects[0]; t = 1 - 1/(1 + (zi[0]-z)/max(1.0, abs(zi[-1]-zi[0]) or 1.0))
            return r.top() + t*r.height()*0.9
        if z >= zi[-1]:
            r = rects[-1]; t = 1 - 1/(1 + (z-zi[-1])/max(1.0, abs(zi[-1]-zi[0]) or 1.0))
            return r.bottom() - t*r.height()*0.9
        k = int(np.searchsorted(zi, z, side='right'))
        r = rects[k]; a, b = zi[k-1], zi[k]
        return r.bottom() - (z-a)/(b-a)*r.height()
