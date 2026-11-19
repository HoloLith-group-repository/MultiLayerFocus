
# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Scientific data files, material catalogue and application assets
datas = (
    collect_data_files('stratified_focus')
    + collect_data_files('multilayerfocus')
)

# Dynamically imported modules
hiddenimports = [
    'vtkmodules.vtkRenderingOpenGL2',
    'vtkmodules.vtkRenderingVolumeOpenGL2',
    'vtkmodules.vtkInteractionStyle',
    'vtkmodules.qt.QVTKRenderWindowInteractor',
    'scipy.io',
    'scipy.special',
    'scipy.interpolate',
    'PIL.Image',
]

# PyQtGraph modules required by interactive plots and exporters
hiddenimports += collect_submodules(
    'pyqtgraph',
    filter=lambda name: (
        'examples' not in name
        and 'opengl' not in name
        and 'jupyter' not in name
    )
)

a = Analysis(
    ['launch_multilayerfocus.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'PyQt5',
        'PySide2',
        'PySide6',
        'tkinter',
        'IPython',
        'pytest',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MultiLayerFocus',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['multilayerfocus/icon.png'],
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='MultiLayerFocus',
)
