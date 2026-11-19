"""CPU ray-cast volume rendering with physical coordinates and common display limits."""
import numpy as np
from matplotlib import colormaps, font_manager, rcParams
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPiecewiseFunction, vtkPolyData, vtkCellArray
from vtkmodules.vtkRenderingCore import (vtkRenderer, vtkVolume, vtkVolumeProperty,
    vtkColorTransferFunction, vtkActor, vtkPolyDataMapper, vtkWindowToImageFilter, vtkTextActor, vtkBillboardTextActor3D)
from vtkmodules.vtkRenderingVolume import vtkFixedPointVolumeRayCastMapper
from vtkmodules.vtkFiltersModeling import vtkOutlineFilter
from vtkmodules.vtkFiltersSources import vtkPlaneSource, vtkCubeSource, vtkLineSource, vtkConeSource
from vtkmodules.vtkRenderingAnnotation import vtkCubeAxesActor, vtkScalarBarActor
import vtkmodules.vtkRenderingOpenGL2
import vtkmodules.vtkRenderingVolumeOpenGL2
import vtkmodules.vtkInteractionStyle


def separate_axis_labels(positions, width, height):
    """Keep short axis names apart in display coordinates, with room at view edges."""
    positions = np.asarray(positions,float)
    placed = []
    for x,y in positions:
        candidates = [(x+dx,y+dy) for dy in (0,32,-32,64,-64,96,-96)
                      for dx in (0,48,-48,96,-96)]
        for px,py in candidates:
            px = float(np.clip(px,24,max(24,width-24)))
            py = float(np.clip(py,24,max(24,height-24)))
            if all(abs(px-qx) >= 44 or abs(py-qy) >= 26 for qx,qy in placed):
                placed.append((px,py)); break
        else:
            # Extremely small viewports cannot contain three full labels.
            placed.append((24+len(placed)*44,24))
    return np.asarray(placed)


def dashed_slice_outline(xmin, xmax, ymin, ymax, z, dashes=16):
    """Build a dashed rectangle as actual line segments for all rendering backends."""
    corners = np.array(((xmin,ymin,z),(xmax,ymin,z),(xmax,ymax,z),(xmin,ymax,z)))
    points = vtkPoints(); lines = vtkCellArray()
    for edge in range(4):
        start, end = corners[edge],corners[(edge+1)%4]
        for dash in range(dashes):
            first = points.InsertNextPoint(*(start+(end-start)*dash/dashes))
            last = points.InsertNextPoint(*(start+(end-start)*(dash+.6)/dashes))
            lines.InsertNextCell(2); lines.InsertCellPoint(first); lines.InsertCellPoint(last)
    data = vtkPolyData(); data.SetPoints(points); data.SetLines(lines)
    return data


class VolumeScene:
    def __init__(self):
        self.font_file = font_manager.findfont(font_manager.FontProperties(family=rcParams['font.family']))
        self.renderer = vtkRenderer(); self.renderer.SetBackground(1, 1, 1)
        self.mapper = vtkFixedPointVolumeRayCastMapper()
        self.mapper.SetBlendModeToComposite()
        self.mapper.AutoAdjustSampleDistancesOff()
        self.mapper.SetLockSampleDistanceToInputSpacing(False)
        self.mapper.SetImageSampleDistance(1.)
        self.mapper.SetMinimumImageSampleDistance(1.)
        self.mapper.SetMaximumImageSampleDistance(1.)
        self.property = vtkVolumeProperty(); self.property.SetInterpolationTypeToLinear()
        self.property.ShadeOff()
        self.actor = vtkVolume(); self.actor.SetMapper(self.mapper); self.actor.SetProperty(self.property)
        self.renderer.AddVolume(self.actor)
        self.outline = vtkOutlineFilter()
        outline_mapper = vtkPolyDataMapper(); outline_mapper.SetInputConnection(self.outline.GetOutputPort())
        self.outline_actor = vtkActor(); self.outline_actor.SetMapper(outline_mapper)
        self.outline_actor.GetProperty().SetColor(.65, .70, .76)
        self.outline_actor.GetProperty().SetOpacity(.18)
        self.outline_actor.GetProperty().SetLineWidth(1.)
        self.renderer.AddActor(self.outline_actor)
        self.axes = vtkCubeAxesActor(); self.axes.SetCamera(self.renderer.GetActiveCamera())
        self.axes.SetFlyModeToClosestTriad()
        for axis in 'XYZ':
            getattr(self.axes, f'Set{axis}Title')(f'{axis.lower()} [µm]')
            getattr(self.axes, f'{axis}AxisLabelVisibilityOff')()
            getattr(self.axes, f'{axis}AxisTickVisibilityOff')()
            getattr(self.axes, f'{axis}AxisMinorTickVisibilityOff')()
            lines = getattr(self.axes, f'Get{axis}AxesLinesProperty')()
            lines.SetColor(.32, .38, .46); lines.SetLineWidth(1.5)
        for index in range(3):
            title = self.axes.GetTitleTextProperty(index)
            title.SetColor(.25,.30,.36); title.SetFontSize(11); title.BoldOff()
        self.renderer.AddActor(self.axes)
        # Explicit billboard names do not depend on CubeAxes' label visibility.
        self.direction_markers = []
        for axis in range(3):
            label = vtkTextActor(); label.SetInput('+'+'xyz'[axis])
            label.GetPositionCoordinate().SetCoordinateSystemToDisplay()
            label.GetTextProperty().SetColor(.25,.30,.36)
            label.GetTextProperty().SetFontSize(12); label.GetTextProperty().BoldOff()
            self._set_font(label.GetTextProperty())
            cone = vtkConeSource(); cone.SetResolution(20)
            direction = [0.,0.,0.]; direction[axis] = 1.; cone.SetDirection(*direction)
            mapper = vtkPolyDataMapper(); mapper.SetInputConnection(cone.GetOutputPort())
            actor = vtkActor(); actor.SetMapper(mapper)
            actor.GetProperty().SetColor(.32,.38,.46); actor.GetProperty().LightingOff()
            self.renderer.AddViewProp(label); self.renderer.AddActor(actor)
            self.direction_markers.append((label,cone,actor))
        self.slice_plane = vtkPlaneSource()
        plane_mapper = vtkPolyDataMapper(); plane_mapper.SetInputConnection(self.slice_plane.GetOutputPort())
        self.slice_actor = vtkActor(); self.slice_actor.SetMapper(plane_mapper)
        self.slice_actor.GetProperty().SetColor(.55, .55, .55)
        self.slice_actor.GetProperty().SetOpacity(.06)
        self.slice_actor.GetProperty().LightingOff()
        self.slice_border_mapper = vtkPolyDataMapper()
        self.slice_border_data = vtkPolyData()
        self.slice_border_mapper.SetInputData(self.slice_border_data)
        self.slice_border = vtkActor(); self.slice_border.SetMapper(self.slice_border_mapper)
        self.slice_border.GetProperty().SetRepresentationToWireframe()
        self.slice_border.GetProperty().SetColor(.38, .38, .38)
        self.slice_border.GetProperty().SetLineWidth(1.5)
        self.slice_border.GetProperty().LightingOff()
        for actor in (self.slice_actor, self.slice_border):
            actor.VisibilityOff(); self.renderer.AddActor(actor)
        self.colorbar = vtkScalarBarActor(); self.colorbar.SetTitle('')
        self.colorbar.SetNumberOfLabels(5)
        self.colorbar.SetWidth(.13); self.colorbar.SetHeight(.45)
        self.colorbar.SetPosition(.82, .27)
        self.colorbar.SetMaximumWidthInPixels(90)
        self.colorbar.SetMaximumHeightInPixels(260)
        self.colorbar.SetBarRatio(.22)
        self.colorbar.SetUnconstrainedFontSize(True)
        self.colorbar.SetLabelFormat('%.3g')
        self.colorbar.GetTitleTextProperty().SetColor(.1, .1, .1)
        self.colorbar.GetLabelTextProperty().SetColor(.1, .1, .1)
        self.colorbar.GetTitleTextProperty().SetFontSize(10)
        self.colorbar.GetLabelTextProperty().SetFontSize(10)
        self.colorbar.GetTitleTextProperty().BoldOff()
        self.colorbar.GetLabelTextProperty().BoldOff()
        self.colorbar.VisibilityOff()
        self.renderer.AddViewProp(self.colorbar)
        self.colorbar_title = vtkTextActor(); self.colorbar_title.SetInput('intensity')
        self.colorbar_title.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self.colorbar_title.SetPosition(.975,.50)
        text = self.colorbar_title.GetTextProperty()
        text.SetOrientation(90); text.SetFontSize(11); text.SetColor(.1,.1,.1)
        text.SetJustificationToCentered(); text.SetVerticalJustificationToCentered()
        self.colorbar_title.VisibilityOff()
        self.renderer.AddViewProp(self.colorbar_title)
        for index in range(3):
            self._set_font(self.axes.GetTitleTextProperty(index))
            self._set_font(self.axes.GetLabelTextProperty(index))
        self._set_font(self.colorbar.GetTitleTextProperty())
        self._set_font(self.colorbar.GetLabelTextProperty())
        self._set_font(self.colorbar_title.GetTextProperty())
        self.image = None
        self.stack_actors = []
        self.stack_bounds = []
        self.layer_markers = []
        self.renderer.GetActiveCamera().AddObserver('ModifiedEvent',self._position_layer_markers)
        self.renderer.AddObserver('StartEvent',self._position_layer_markers)

    def _set_font(self, text_property):
        text_property.SetFontFamilyAsString('File')
        text_property.SetFontFile(self.font_file)

    def set_data(self, values, x, y, z, reset=False):
        values = np.asarray(values, np.float32)
        if values.shape != (len(z), len(y), len(x)):
            raise ValueError('Volume scalar grid and coordinate lengths differ')
        self.image = vtkImageData()
        self.image.SetDimensions(len(x), len(y), len(z))
        self.image.SetOrigin(float(x[0]), float(y[0]), float(z[0]))
        spacing = tuple(float(v[1]-v[0]) for v in (x, y, z))
        self.image.SetSpacing(*spacing)
        self.image.GetPointData().SetScalars(numpy_to_vtk(values.ravel(order='C'), deep=True))
        self.mapper.SetInputData(self.image); self.outline.SetInputData(self.image)
        self.axes.SetBounds(self.image.GetBounds())
        self._position_layer_markers()
        # Coordinates are in micrometres: VTK's default 1/2-unit ray steps can
        # skip an entire focal volume. Keep physical sampling fixed while rotating.
        self.mapper.SetSampleDistance(.5*min(spacing))
        self.mapper.SetInteractiveSampleDistance(.5*min(spacing))
        spans = [float(v[-1]-v[0]) for v in (x,y,z)]
        self.property.SetScalarOpacityUnitDistance(max(spans)/10)
        if reset:
            self.reset_camera()

    def set_slice_plane(self, z, visible=True, opacity=.06):
        """Mark an XY section in the same physical coordinates as the volume."""
        if self.image is None:
            return
        if not np.isfinite(opacity) or not 0 <= opacity <= 1:
            raise ValueError('Slice plane opacity must lie between zero and one')
        self.slice_actor.GetProperty().SetOpacity(float(opacity))
        xmin, xmax, ymin, ymax, zmin, zmax = self.image.GetBounds()
        tolerance = 1e-12*max(1.,abs(zmin),abs(zmax))
        if not np.isfinite(z) or z < zmin-tolerance or z > zmax+tolerance:
            raise ValueError('Slice position must lie within the calculated volume')
        # Image bounds reconstructed from spacing may differ by roundoff at endpoints.
        z = float(np.clip(z,zmin,zmax))
        self.slice_plane.SetOrigin(xmin, ymin, z)
        self.slice_plane.SetPoint1(xmax, ymin, z)
        self.slice_plane.SetPoint2(xmin, ymax, z)
        self.slice_plane.Update()
        self.slice_border_data = dashed_slice_outline(xmin,xmax,ymin,ymax,z)
        self.slice_border_mapper.SetInputData(self.slice_border_data)
        self.slice_actor.SetVisibility(visible)
        self.slice_border.SetVisibility(visible)

    def clear_stack(self):
        for actor in self.stack_actors:
            self.renderer.RemoveActor(actor)
        for line, actor, label, _ in self.layer_markers:
            self.renderer.RemoveActor(actor)
            if label is not None: self.renderer.RemoveActor(label)
        self.layer_markers = []
        self.stack_actors = []; self.stack_bounds = []

    def set_stack(self, stack, visible=False, opacity=.035):
        """Show layer slabs clipped to the calculated volume, including end media."""
        self.clear_stack()
        if self.image is None:
            return
        xmin, xmax, ymin, ymax, zmin, zmax = self.image.GetBounds()
        interfaces = stack.interfaces if len(stack.indices) > 1 else np.array([])
        starts = np.r_[-np.inf, interfaces]; ends = np.r_[interfaces, np.inf]
        for index, (start, end) in enumerate(zip(starts, ends)):
            low, high = max(float(start),zmin), min(float(end),zmax)
            if high <= low:
                continue
            bounds = (xmin,xmax,ymin,ymax,low,high)
            cube = vtkCubeSource(); cube.SetBounds(bounds)
            mapper = vtkPolyDataMapper(); mapper.SetInputConnection(cube.GetOutputPort())
            self.stack_bounds.append((index,bounds))
            for wireframe in (False,True):
                actor = vtkActor(); actor.SetMapper(mapper)
                prop = actor.GetProperty(); prop.SetColor(*map(float,colormaps['Pastel1'](index % 9)[:3]))
                prop.LightingOff()
                if wireframe:
                    prop.SetRepresentationToWireframe(); prop.SetLineWidth(1.)
                prop.SetOpacity(min(1.,opacity+.25) if wireframe else opacity)
                actor.SetVisibility(visible)
                self.renderer.AddActor(actor); self.stack_actors.append(actor)
            n = complex(stack.indices[index])
            text = f'n = {n.real:g}' + (f' {n.imag:+g}i' if n.imag else '')
            self._add_layer_marker((low+high)/2,text,colormaps['Pastel1'](index % 9)[:3])
        self._position_layer_markers()
        self.set_stack_opacity(opacity)

    def _add_layer_marker(self, z, text, color):
        line = vtkLineSource(); mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(line.GetOutputPort())
        actor = vtkActor(); actor.SetMapper(mapper)
        actor.GetProperty().SetColor(.32,.38,.46); actor.GetProperty().SetLineWidth(1.5)
        actor.VisibilityOff()
        self.renderer.AddActor(actor)
        label = None
        if text is not None:
            label = vtkBillboardTextActor3D(); label.SetInput(text)
            label.GetTextProperty().SetColor(*map(float,color))
            label.GetTextProperty().SetFontSize(15); label.GetTextProperty().BoldOn()
            label.GetTextProperty().SetOpacity(1.)
            self._set_font(label.GetTextProperty())
            self.renderer.AddActor(label)
        self.layer_markers.append((line,actor,label,z))

    def _position_layer_markers(self, *_):
        if self.image is None:
            return
        xmin,xmax,ymin,ymax,zmin,zmax = self.image.GetBounds()
        camera = np.array(self.renderer.GetActiveCamera().GetPosition())
        corners = [(x,y) for x in (xmin,xmax) for y in (ymin,ymax)]
        x,y = min(corners,key=lambda c:(c[0]-camera[0])**2+(c[1]-camera[1])**2)
        zside = min((zmin,zmax),key=lambda z:abs(z-camera[2]))
        corner = np.array([x,y,zside])
        maxima = np.array([xmax,ymax,zmax])
        spans = np.array([xmax-xmin,ymax-ymin,zmax-zmin])
        arrow_size = .05*min(spans)
        anchors = []; axis_centers = []
        for axis,(label,cone,actor) in enumerate(self.direction_markers):
            end = corner.copy(); end[axis] = maxima[axis]
            center = end.copy(); center[axis] -= arrow_size/2
            cone.SetCenter(*center); cone.SetHeight(arrow_size); cone.SetRadius(arrow_size/4)
            position = corner.copy()
            position[axis] = (self.image.GetBounds()[2*axis]+maxima[axis])/2
            axis_centers.append(position.copy())
            # Names follow separate edge midpoints rather than their shared corner.
            for other in range(3):
                if other != axis:
                    position[other] += (.04 if corner[other] == maxima[other] else -.04)*spans[other]
            anchors.append(position)
        width,height = self.renderer.GetSize()
        if width > 0 and height > 0:
            projected = []
            for axis,position in enumerate(anchors):
                self.renderer.SetWorldPoint(*position,1.)
                self.renderer.WorldToDisplay()
                outside = np.array(self.renderer.GetDisplayPoint()[:2])
                self.renderer.SetWorldPoint(*axis_centers[axis],1.)
                self.renderer.WorldToDisplay()
                center = np.array(self.renderer.GetDisplayPoint()[:2])
                direction = outside-center
                length = np.linalg.norm(direction)
                direction = direction/length if length > 1e-9 else np.array([-1.,0.])
                projected.append(center+direction*(34 if axis == 2 else 26))
            placed = separate_axis_labels(projected,width,height)
        else:
            placed = np.array([(24,24),(80,24),(24,80)])
        for (label,_,_),position in zip(self.direction_markers,placed):
            label.SetPosition(*position)
        dx = (1 if x == xmax else -1)*.035*(xmax-xmin)
        dy = (1 if y == ymax else -1)*.035*(ymax-ymin)
        for line,actor,label,z in self.layer_markers:
            line.SetPoint1(x,y,z); line.SetPoint2(x+dx,y+dy,z)
            if label is not None: label.SetPosition(x+2*dx,y+2*dy,z)

    def set_stack_opacity(self, opacity):
        if not np.isfinite(opacity) or not 0 <= opacity <= 1:
            raise ValueError('Stack opacity must lie between zero and one')
        for index,actor in enumerate(self.stack_actors):
            actor.GetProperty().SetOpacity(min(1.,opacity+.25) if index % 2 else opacity)

    def set_stack_visible(self, visible):
        for actor in self.stack_actors:
            actor.SetVisibility(visible)

    def set_transfer(self, cmap, threshold=.02, gamma=1.6, alpha_max=.30, high=1.):
        if not np.isfinite(high) or high <= 0:
            raise ValueError('Color scale maximum must be positive and finite')
        colors = vtkColorTransferFunction(); opacity = vtkPiecewiseFunction()
        for value in np.linspace(0, 1, 256):
            colors.AddRGBPoint(float(value*high), *map(float, cmap(value)[:3]))
            opacity.AddPoint(float(value*high), float(alpha_max*value**gamma if value >= threshold else 0))
        self.property.SetColor(colors); self.property.SetScalarOpacity(opacity)
        self.colorbar.SetLookupTable(colors)

    def camera_angles(self):
        """Return camera-position azimuth/elevation in degrees about its focal point."""
        camera = self.renderer.GetActiveCamera()
        delta = np.array(camera.GetPosition())-np.array(camera.GetFocalPoint())
        return (float(np.degrees(np.arctan2(delta[1],delta[0]))),
                float(np.degrees(np.arctan2(delta[2],np.hypot(delta[0],delta[1])))))

    def set_camera_angles(self, azimuth, elevation):
        """Turn around +z with bounded elevation, preserving the focal point, distance and zoom."""
        if not np.isfinite(azimuth) or not np.isfinite(elevation) or not -89 <= elevation <= 89:
            raise ValueError('Camera angles must be finite; elevation must lie between -89 and 89 degrees')
        camera = self.renderer.GetActiveCamera()
        focal = np.array(camera.GetFocalPoint())
        distance = max(float(np.linalg.norm(np.array(camera.GetPosition())-focal)),1e-12)
        az,el = np.radians((azimuth,elevation))
        direction = np.array((np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)))
        camera.SetPosition(*(focal+distance*direction)); camera.SetViewUp(0,0,1)
        self.renderer.ResetCameraClippingRange()

    def reset_camera(self):
        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(0,0,0)
        camera.SetPosition(np.sqrt(3/8),-np.sqrt(3/8),.5)
        camera.SetViewUp(0,0,1)
        self.renderer.ResetCamera(); self.renderer.ResetCameraClippingRange()


def make_turntable_style(scene):
    """Use azimuth/elevation mouse rotation with wheel zoom for a z-up scene."""
    from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTerrain
    style = vtkInteractorStyleTerrain()
    def wheel(_,event):
        scene.renderer.GetActiveCamera().Dolly(1.1 if event == 'MouseWheelForwardEvent' else 1/1.1)
        scene.renderer.ResetCameraClippingRange()
        style.InvokeEvent('InteractionEvent')
        interactor = style.GetInteractor()
        if interactor is not None: interactor.Render()
    style.AddObserver('MouseWheelForwardEvent',wheel)
    style.AddObserver('MouseWheelBackwardEvent',wheel)
    return style


def capture_image(window):
    """Return an RGB Pillow image of the current camera view."""
    from PIL import Image
    window.Render()
    grab = vtkWindowToImageFilter(); grab.SetInput(window)
    grab.SetInputBufferTypeToRGB(); grab.ReadFrontBufferOff(); grab.Update()
    output = grab.GetOutput(); width, height, _ = output.GetDimensions()
    pixels = vtk_to_numpy(output.GetPointData().GetScalars()).reshape(height, width, 3)
    return Image.fromarray(pixels[::-1].copy())
