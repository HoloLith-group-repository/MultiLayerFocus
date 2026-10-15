# Run in the project folder using the activated virtual environment.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$buildArgs = @(
    '--noconfirm', '--clean', '--onedir', '--windowed',
    '--name', 'MultiLayerFocus',
    '--collect-data', 'stratified_focus',
    '--add-data', 'stratified_focus/colormaps.npz:stratified_focus',
    '--hidden-import', 'vtkmodules.vtkRenderingOpenGL2',
    '--hidden-import', 'vtkmodules.vtkRenderingVolumeOpenGL2',
    '--hidden-import', 'vtkmodules.vtkInteractionStyle',
    '--hidden-import', 'vtkmodules.qt.QVTKRenderWindowInteractor',
    '--exclude-module', 'PyQt5', '--exclude-module', 'PySide2', '--exclude-module', 'PySide6'
)
$softwareIcon = if (Test-Path 'stratified_focus/icon.png') {
    'stratified_focus/icon.png'
} elseif (Test-Path 'icon.png') {
    'icon.png'
} else {
    $null
}
if ($softwareIcon) {
    $buildArgs += @('--add-data', "${softwareIcon}:stratified_focus", '--icon', $softwareIcon)
} else {
    Write-Host 'icon.png is absent; building without a custom icon.'
}
$buildArgs += 'launch_gui.py'
& python -m PyInstaller @buildArgs
exit $LASTEXITCODE
