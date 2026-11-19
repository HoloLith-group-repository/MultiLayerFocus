# MultiLayerFocus

<p align="center">
  <img src="logo.png" alt="MultiLayerFocus logo" width="300">
</p>

<p align="center">
  <strong>A toolbox for vectorial focusing in stratified media</strong>
</p>

**MultiLayerFocus** is an open-source Python-based numerical toolbox for simulating high-numerical-aperture (NA) focusing in homogeneous and stratified optical media.

The software combines the vectorial **Debye–Wolf diffraction formalism** with a coherent **multilayer transfer-matrix approach** to model the propagation of tightly focused electromagnetic fields through planar stratified structures.

It enables the reconstruction and analysis of the complex electric field before, **inside**, and after multilayer structures, accounting for polarization-dependent transmission, reflection, multiple interference, and absorption.

MultiLayerFocus provides a graphical user interface (GUI) for configuring optical systems, performing numerical simulations, analyzing vectorial electromagnetic fields, visualizing three-dimensional distributions, and exporting scientific results.

## Key Features

### Optical Simulation

- **Vectorial high-NA focusing:** Numerical reconstruction of the complex Cartesian electric-field components (Ex, Ey, Ez) using the Debye–Wolf formalism.
- **Multilayer optical structures:** Simulation of dielectric, absorbing, and complex-refractive-index media.
- **Multiple reflections:** Coherent treatment of forward- and backward-propagating electromagnetic fields within finite layers.
- **Numerical integration methods:** Support for trapezoidal, Gauss–Legendre, and Fourier–Bessel integration strategies.
- **Flexible pupil illumination:** Uniform, linear, elliptical, circular, radial, azimuthal, and mixed cylindrical-vector polarization.
- **Structured optical fields:** Configurable vortex phase and Zernike aberrations.
- **Custom pupil fields:** Import of externally generated complex pupil distributions.

### Field Reconstruction and Analysis

- Transverse XY field reconstruction.
- Longitudinal XZ and YZ field reconstruction.
- Three-dimensional XYZ volumetric reconstruction.
- Individual electric-field components and total intensity.
- Three-dimensional polarization ellipse analysis.
- Electromagnetic energy flow and Poynting-vector analysis.
- Absorption and multilayer energy-budget analysis.
- Automatic focal-field metrics, including FWHM.
- Parameter-sweep analysis.

### Graphical Interface and Visualization

- Interactive configuration of objective, illumination, multilayer geometry, and observation parameters.
- Live Preview for rapid inspection of simulation results.
- Adjustable spatial and angular sampling.
- Interactive two-dimensional field maps.
- Light and Dark interface themes.
- Slice-by-slice visualization of reconstructed volumetric fields.
- Global or slice-dependent intensity scaling.
- Automatic best-focus visualization.
- Interactive three-dimensional volume rendering with configurable visualization settings.
- Independent rendering resolution for the 3D visualizer.
- Scientific colorbars and configurable display parameters.

### Scientific Data Export

- Export of individual simulation tabs in TXT or MATLAB MAT format.
- Complete export of all available tabs in both TXT and MAT formats.
- Organized export directories with customizable names.
- Export of all volumetric XY slices at the calculated axial coordinates.
- Numerical coordinate grids and simulation metadata.
- Graphical export and detailed simulation reports.
- JSON-based optical configuration saving and loading.

## Scientific Background

High-NA focusing requires a vectorial description of electromagnetic propagation, particularly when the focal region contains refractive-index discontinuities.

MultiLayerFocus combines two complementary theoretical approaches:

1. **Debye–Wolf diffraction theory**, describing the coherent superposition of angular plane-wave components focused by a high-NA objective.

2. **Multilayer transfer-matrix methods**, describing polarization-dependent reflection, transmission, and interference in planar stratified optical media.

The resulting framework provides access to the complex electromagnetic field throughout the optical structure, including internal reflections, longitudinal polarization components, and the field distributions within absorbing layers.

The numerical formulation, validation, and representative optical applications are described in the accompanying scientific manuscript:

*MultiLayerFocus – a toolbox for vectorial focusing in stratified media.*

## Installation and Execution

### Requirements

MultiLayerFocus is developed in Python and supports execution through its graphical interface or direct use of its numerical modules.

Dependencies are specified in the repository's `requirements.txt` and `pyproject.toml` files.

### Run from source

Clone or download the repository and open a terminal in the project directory containing `pyproject.toml`.

Install the package and its dependencies:

```bash
python -m pip install .
```

Alternatively, install the runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

Launch the graphical interface:

```bash
python launch_multilayerfocus.py
```

For additional information, consult the MultiLayerFocus GUI User Guide.

### Windows Portable Application

Precompiled Windows distributions, when available, can be downloaded from the **Releases** section of the GitHub repository.

The Windows version is distributed as a portable application.

To run MultiLayerFocus:

1. Download the Windows distribution ZIP.
2. Extract the complete archive into a folder.
3. Open the extracted `MultiLayerFocus` directory.
4. Run `MultiLayerFocus.exe`.

**Important:** Keep the executable together with its accompanying files and the `_internal` directory.

A separate Python installation is not required to use the compiled Windows application.

## Getting Started

A typical simulation follows these steps:

1. **Configure the objective:** Specify the numerical aperture, focal length, immersion medium, and relevant optical parameters.

2. **Define the illumination:** Select the wavelength, pupil amplitude, polarization state, and optional vortex phase or Zernike aberrations.

3. **Configure the multilayer:** Specify the refractive indices and thicknesses of the optical layers, ordered from the incident medium to the exit medium.

4. **Set the observation region:** Define the spatial coordinates, observation window, and sampling resolution.

5. **Choose numerical accuracy:** Configure angular sampling, numerical integration settings, and optional full-volume reconstruction.

6. **Run the simulation:** Use the **Compute** button to calculate the electromagnetic field, or enable **Live Preview** for rapid inspection.

7. **Analyze results:** Navigate between the simulation tabs to inspect electromagnetic fields, polarization, energy flow, volumetric distributions, and focal properties.

8. **Export results:** Save individual tab data or export all available simulation tabs in TXT and MATLAB MAT formats.

## Simulation Results

Depending on the selected configuration, MultiLayerFocus provides:

- Complex Cartesian electric-field components Ex, Ey, and Ez.
- Total and component-resolved electric-field intensities.
- Transverse and longitudinal intensity distributions.
- Three-dimensional focal-field reconstruction.
- Focal-spot metrics and FWHM analysis.
- Spatially resolved polarization ellipse reconstruction.
- Electromagnetic energy flow and absorption.
- Reflection, transmission, and absorption properties of multilayer structures.

The graphical interface organizes these results into dedicated tabs:

- **Pupil Plane**
- **Stack Response**
- **Focal Field**
- **Focal Field Components**
- **Polarization**
- **Energy**
- **Full Volume Reconstruction**
- **Parameter Sweep**

### Full Volume Reconstruction

The Full Volume Reconstruction tab allows the calculation and analysis of the electromagnetic field over a three-dimensional spatial domain.

Users can select the transverse and axial sampling resolution and inspect the resulting field through XY slices, axial sections, and intensity projections.

The interface provides:

- Interactive axial slice selection.
- Normalized intensity visualization.
- Local intensity rescaling for improved visibility of weak field distributions.
- Automatic best-focus identification.
- Interactive 3D volume visualization.

The 3D renderer uses a separate visualization grid to manage graphical performance, without altering the sampling resolution of the underlying scientific calculation.

The calculated volumetric data retain the resolution specified by the user.

### Volumetric Sampling

The graphical interface supports a minimum volumetric sampling of **10 × 10 × 10** points.

The total number of calculated voxels is limited to **2,000,000**, providing a balance between numerical flexibility and computational resource requirements.

Computational time and memory requirements depend on the selected spatial and angular sampling resolutions.

Users should verify numerical convergence when performing quantitative scientific analyses.

## Data Export

MultiLayerFocus supports both individual-tab and complete-session data export.

### Individual Tab Export

The **Export** menu provides:

- Figure of Current Tab (PNG)
- Current Tab Data (TXT)
- Current Tab Data (MAT)
- Detailed Simulation Report (PDF)

Numerical exports include the relevant field data, coordinate information, and simulation metadata.

The destination directory and output folder name can be selected by the user.

### Export All Tabs

The command **Export All Tabs (TXT + MAT)** saves the available numerical results from all simulation tabs in a single operation.

Each tab is organized in a dedicated subdirectory, with separate TXT and MAT folders.

Example structure:

```text
MultiLayerFocus_Complete_Export/
├── Pupil_Plane/
│   ├── TXT/
│   └── MAT/
├── Stack_Response/
│   ├── TXT/
│   └── MAT/
├── Focal_Field/
│   ├── TXT/
│   └── MAT/
├── Focal_Field_Components/
│   ├── TXT/
│   └── MAT/
├── Polarization/
│   ├── TXT/
│   └── MAT/
├── Energy/
│   ├── TXT/
│   └── MAT/
├── Full_Volume_Reconstruction/
│   ├── TXT/
│   └── MAT/
└── Parameter_Sweep/
    ├── TXT/
    └── MAT/
```

Tabs without available calculated results may be skipped.

### Volumetric Slice Export

When full-volume data are available, MultiLayerFocus additionally exports a `slices` directory containing the XY intensity distribution for each calculated axial coordinate.

The output format follows the selected export type.

Example:

```text
Full_Volume_Reconstruction/
├── TXT/
│   └── slices/
│       ├── x.txt
│       ├── y.txt
│       ├── slice_z=-0.5.txt
│       ├── slice_z=0.txt
│       └── slice_z=0.5.txt
└── MAT/
    └── slices/
        ├── x.mat
        ├── y.mat
        ├── slice_z=-0.5.mat
        ├── slice_z=0.mat
        └── slice_z=0.5.mat
```

The numerical matrices are stored using the original calculated spatial sampling, independently of the visualization resolution used by the 3D renderer.

Slice intensities are normalized with respect to the global maximum intensity of the reconstructed volume.

This preserves the relative intensity distribution between different axial planes.

## Documentation

Detailed instructions for configuring and operating MultiLayerFocus are provided in the **MultiLayerFocus GUI User Guide**.

The guide covers:

- Optical objective parameters.
- Illumination and polarization configuration.
- Zernike aberrations and vortex phases.
- External pupil-field import.
- Multilayer optical structures.
- Numerical integration methods.
- Spatial and angular sampling.
- Field and polarization analysis.
- Volumetric visualization.
- Scientific data export.

## Applications

MultiLayerFocus can support numerical investigations in:

- High-resolution and vectorial microscopy.
- Optical trapping and optical manipulation.
- Structured light and polarization engineering.
- Focusing through dielectric and absorbing multilayers.
- Laser nanofabrication and direct laser writing.
- Optical field reconstruction and photonic simulations.

## Authors

MultiLayerFocus is associated with the scientific work of:

- **Francesco Reda**
- **Davide Romano**
- **Marcella Salvatore**
- **Marco Piccardo**
- **Stefano Luigi Oscurato**

**Affiliations:**

Department of Physics “E. Pancini”, University of Naples Federico II, Naples, Italy.

Department of Physics, Instituto Superior Técnico, Universidade de Lisboa, Lisbon, Portugal.

Instituto de Engenharia de Sistemas e Computadores – Microsistemas e Nanotecnologias (INESC MN), Lisbon, Portugal.

**Corresponding authors:**

- Francesco Reda — francesco.reda@unina.it
- Stefano Luigi Oscurato — stefanoluigi.oscurato@unina.it

## Citation

If you use MultiLayerFocus in scientific research, please cite the associated scientific publication:

**Reda, F., Romano, D., Salvatore, M., Piccardo, M., and Oscurato, S. L.**

*MultiLayerFocus – a toolbox for vectorial focusing in stratified media.*

Open Research Europe (2026).

Please refer to the `CITATION.cff` file for the preferred citation format and updated bibliographic information.

When distributing or adapting material under CC BY 4.0, the attribution requirements of the license must also be respected.

## License

**Copyright © 2026**

Francesco Reda, Davide Romano, Marcella Salvatore, Marco Piccardo, and Stefano Luigi Oscurato.

MultiLayerFocus is distributed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**, as specified in the accompanying `LICENSE` file.

Subject to the license conditions, users may:

- Copy and redistribute the licensed material.
- Adapt, modify, and build upon the licensed material.
- Use the licensed material for academic, educational, research, and commercial purposes.

Appropriate attribution must be provided, including identification of the authors, reference to the license, and indication of modifications when applicable.

The CC BY 4.0 license does not impose a non-commercial restriction.

Third-party libraries and dependencies remain subject to their respective licenses.

See the `LICENSE` file for the complete license terms
