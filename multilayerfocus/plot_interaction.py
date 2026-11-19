
"""Shared mouse controls and context menu for MultiLayerFocus plots."""

import csv
import numpy as np
import pyqtgraph as pg

from pyqtgraph.exporters import ImageExporter, SVGExporter
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFileDialog, QMenu, QMessageBox


class UnifiedViewBox(pg.ViewBox):
    """Rectangle zoom and a shared context menu for all 2D plots."""

    def __init__(self):
        super().__init__()

        self.plot_item = None
        self.data_owner = None
        self.grid_visible = False
        self.axes_visible = True
        self._home_range = None
        self._axes_before_hide = None

        # Left-button rectangle zoom
        self.setMouseMode(pg.ViewBox.RectMode)

    def remember_home(self):
        self._home_range = [
            list(axis_range) for axis_range in self.viewRange()
        ]

    def forget_home(self):
        self._home_range = None

    def reset_view(self):
        if self._home_range is None:
            self.autoRange(padding=0.02)
        else:
            self.setRange(
                xRange=self._home_range[0],
                yRange=self._home_range[1],
                padding=0
            )

    def wheelEvent(self, ev, axis=None):
        # Disable wheel zoom everywhere, including the axes
        ev.accept()

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == Qt.MouseButton.LeftButton and axis is None:
            if ev.isStart() and self._home_range is None:
                self.remember_home()

            # Use PyQtGraph's native rectangle zoom
            super().mouseDragEvent(ev, axis)
        else:
            # Disable pan, right-drag zoom, and middle-drag
            ev.accept()

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton and ev.double():
            self.reset_view()
            ev.accept()

        elif ev.button() == Qt.MouseButton.RightButton:
            ev.accept()
            self._show_menu(ev.screenPos().toPoint())

        else:
            super().mouseClickEvent(ev)

    def _show_menu(self, position):
        if self.plot_item is None:
            return

        menu = QMenu()

        reset = menu.addAction("Reset view")

        menu.addSeparator()

        grid = menu.addAction("Show grid")
        grid.setCheckable(True)
        grid.setChecked(self.grid_visible)

        axes = menu.addAction("Show axes")
        axes.setCheckable(True)
        axes.setChecked(self.axes_visible)

        menu.addSeparator()

        copy_image = menu.addAction("Copy image")
        png = menu.addAction("Export PNG...")
        svg = menu.addAction("Export SVG...")
        data = menu.addAction("Export data...")

        selected = menu.exec(position)

        try:
            if selected == reset:
                self.reset_view()

            elif selected == grid:
                self.grid_visible = grid.isChecked()
                self.plot_item.showGrid(
                    x=self.grid_visible,
                    y=self.grid_visible,
                    alpha=0.18
                )

            elif selected == axes:
                self.axes_visible = axes.isChecked()

                if not self.axes_visible:
                    self._axes_before_hide = {
                        side: self.plot_item.getAxis(side).isVisible()
                        for side in ("left", "bottom")
                    }
                    for side in ("left", "bottom"):
                        self.plot_item.showAxis(side, False)
                else:
                    previous = self._axes_before_hide or {
                        "left": True,
                        "bottom": True
                    }
                    for side in ("left", "bottom"):
                        self.plot_item.showAxis(
                            side, previous[side]
                        )

            elif selected == copy_image:
                ImageExporter(self.plot_item).export(copy=True)

            elif selected == png:
                self._export_image(".png", ImageExporter)

            elif selected == svg:
                self._export_image(".svg", SVGExporter)

            elif selected == data:
                self._export_data()

        except Exception as exc:
            QMessageBox.warning(
                None, "Plot operation", str(exc)
            )

    def _choose_file(self, extension, description):
        path, _ = QFileDialog.getSaveFileName(
            None,
            "Export plot",
            "",
            f"{description} (*{extension})"
        )

        if path and not path.lower().endswith(extension):
            path += extension

        return path

    def _export_image(self, extension, exporter):
        description = (
            "PNG image" if extension == ".png" else "SVG image"
        )

        path = self._choose_file(extension, description)

        if path:
            exporter(self.plot_item).export(path)

    def _export_data(self):
        plot = self.plot_item
        owner = self.data_owner

        # Export the original numerical values of a 2D map
        if (
            owner is not None
            and hasattr(owner, "values")
            and owner.values is not None
        ):
            path = self._choose_file(".csv", "CSV data")
            if not path:
                return

            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                # X coordinates in the first row, Y in the first column
                writer.writerow(
                    ["y / x"] + list(np.asarray(owner.x))
                )

                for y, row in zip(owner.y, owner.values):
                    writer.writerow([y] + list(row))
            return

        # Export numerical curves
        curves = []

        for item in plot.listDataItems():
            if not hasattr(item, "getData"):
                continue

            x, y = item.getData()

            if x is not None and y is not None and len(x):
                curves.append((
                    np.asarray(x),
                    np.asarray(y)
                ))

        # Also support bar graphs
        bars = [
            item for item in plot.items
            if isinstance(item, pg.BarGraphItem)
        ]

        if not curves and not bars:
            QMessageBox.information(
                None,
                "Export data",
                "No numeric data to export."
            )
            return

        path = self._choose_file(".csv", "CSV data")
        if not path:
            return

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if curves:
                header = []

                for i in range(len(curves)):
                    header.extend([
                        f"x_{i + 1}",
                        f"y_{i + 1}"
                    ])

                writer.writerow(header)

                nrows = max(len(x) for x, _ in curves)

                for i in range(nrows):
                    row = []

                    for x, y in curves:
                        if i < len(x):
                            row.extend([x[i], y[i]])
                        else:
                            row.extend(["", ""])

                    writer.writerow(row)

            else:
                writer.writerow([
                    "x0", "y", "width", "height"
                ])

                for item in bars:
                    options = item.opts

                    x0, y, width, height = np.broadcast_arrays(
                        np.asarray(options.get("x0")),
                        np.asarray(options.get("y")),
                        np.asarray(options.get("width")),
                        np.asarray(options.get("height"))
                    )

                    for values in zip(
                        x0.flat,
                        y.flat,
                        width.flat,
                        height.flat
                    ):
                        writer.writerow(values)


def setup_unified_plot(plot, owner=None, grid=False):
    """Attach the common controls to a PyQtGraph PlotItem."""

    viewbox = plot.getViewBox()

    viewbox.plot_item = plot
    viewbox.data_owner = owner
    viewbox.grid_visible = grid

    # Disable the standard PlotItem menu
    # while retaining our custom ViewBox menu
    plot.setMenuEnabled(
        False,
        enableViewBoxMenu=None
    )

    return viewbox
