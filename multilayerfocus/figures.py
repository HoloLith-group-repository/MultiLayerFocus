"""Publication-quality figures and PDF report (matplotlib)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from .engine import quantity


def _interfaces(r):
    st = r.config.stack
    return [] if len(st.indices) == 1 else list(st.interfaces)


def publication_figure(r, path, cmap='inferno'):
    I_xz = np.sum(np.abs(r.Exz*r.scale)**2, 0); I_xy = np.sum(np.abs(r.Exy*r.scale)**2, 0)
    peak = max(I_xz.max(), I_xy.max())
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
    im = ax[0].imshow(I_xz/peak, extent=(r.x[0], r.x[-1], r.z[0], r.z[-1]), origin='lower', cmap=cmap, vmin=0, vmax=1)
    for z in _interfaces(r):
        ax[0].axhline(z, color='w', ls='--', lw=0.8, alpha=0.8)
    ax[0].axhline(r.scene.z0, color='#6b9bff', lw=1.2)
    ax[0].set(xlabel='x [µm]', ylabel='z [µm]', title='|E|² · XZ section')
    ax[1].imshow(I_xy/peak, extent=(r.x[0], r.x[-1], r.y[0], r.y[-1]), origin='lower', cmap=cmap, vmin=0, vmax=1)
    ax[1].set(xlabel='x [µm]', ylabel='y [µm]', title=f'|E|² · XY plane, z₀ = {r.scene.z0:g} [µm]')
    cb = fig.colorbar(im, ax=ax, shrink=0.85); cb.set_label(f'|E|² / {peak:.3g} [V²/m²]  ({r.scene.power_mW:g} [mW])')
    fig.savefig(path, dpi=300)
    plt.close(fig)


def report(r, path):
    s = r.scene; m = r.metrics; b = r.budget
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.95, 'MultiLayerFocus — Simulation Report', fontsize=16, weight='bold', color='#173e61')
        lines = [f'Objective: {s.objective}  ·  NA {s.na:g}  ·  f = {s.focal_length:g} [µm]',
                 f'Wavelength {s.wavelength:g} [µm]  ·  polarization {s.polarization}  ·  vortex ℓ = {s.vortex}  ·  '
                 f'{"uniform pupil" if s.fill <= 0 else f"Gaussian fill {s.fill:g}"}  ·  power {s.power_mW:g} [mW]',
                 'Stack (incident → exit): ' + ' | '.join(f'{l.name.split(" (")[0]} n={l.n_re:g}{l.n_im:+g}i' +
                                                         ('' if l.d is None else f' d={l.d:g} [µm]') for l in s.layers),
                 f'Last interface z_out = {s.z_out:g} [µm]  ·  XY plane z0 = {s.z0:g} [µm]  ·  window {s.lateral:g} × {s.axial:g} [µm]',
                 f'Angular integration: {r.ntheta} Gauss–Legendre polar nodes, {r.method}  ·  estimated relative error '
                 f'{"n/a" if r.sampling_error is None else f"{r.sampling_error:.1e}"}  ·  compute time {r.seconds:.2f} [s]', '',
                 f'Peak |E| = {m["E_peak_Vm"]:.4g} [V/m]  ·  peak intensity {m["I_peak_Wcm2"]:.4g} [W/cm²]  ·  at z = {m["z_peak"]:+.3f} [µm]',
                 f'FWHM x, y, z = {1000*m["fwhm_x"]:.0f}, {1000*m["fwhm_y"]:.0f}, {1000*m["fwhm_z"]:.0f} [nm]  ·  '
                 f'|Ex|²:|Ey|²:|Ez|² = ' + ' : '.join(f'{100*v:.1f}%' for v in m['frac']),
                 f'Beam power budget: R = {100*b["R"]:.2f} %, T = {100*b["T"]:.2f} %, absorbed = '
                 + ', '.join(f'{100*a:.2f} % ({s.layers[j].name.split(" (")[0]})' for j, a in enumerate(b['A']) if a > 1e-6)]
        if r.metrics_ref:
            m0 = r.metrics_ref
            lines.append(f'Aberration correction for z = {s.correct_z:g} [µm]: peak intensity {m0["I_peak_Wcm2"]:.4g} → '
                         f'{m["I_peak_Wcm2"]:.4g} [W/cm²] (× {m["I_peak_Wcm2"]/m0["I_peak_Wcm2"]:.2f}), peak at '
                         f'{m0["z_peak"]:+.3f} → {m["z_peak"]:+.3f} [µm]')
        y = 0.91
        for ln in lines:
            fig.text(0.08, y, ln, fontsize=8.5, wrap=True); y -= 0.022
        ax1 = fig.add_axes([0.08, 0.38, 0.4, 0.28]); ax2 = fig.add_axes([0.55, 0.38, 0.4, 0.28])
        I_xz = np.sum(np.abs(r.Exz)**2, 0); I_xy = np.sum(np.abs(r.Exy)**2, 0); pk = I_xz.max()
        ax1.imshow(I_xz/pk, extent=(r.x[0], r.x[-1], r.z[0], r.z[-1]), origin='lower', cmap='inferno')
        for z in _interfaces(r): ax1.axhline(z, color='w', ls='--', lw=0.8)
        ax1.set(xlabel='x [µm]', ylabel='z [µm]', title='|E|² XZ')
        ax2.imshow(I_xy/I_xy.max(), extent=(r.x[0], r.x[-1], r.y[0], r.y[-1]), origin='lower', cmap='inferno')
        ax2.set(xlabel='x [µm]', ylabel='y [µm]', title='|E|² XY')
        ax3 = fig.add_axes([0.08, 0.06, 0.87, 0.24])
        for c, col in zip(range(3), ('#3366ee', '#2bb673', '#e8590c')):
            ax3.plot(r.x, np.abs(r.Exy[c, len(r.y)//2])**2/I_xy.max(), color=col, label=f'|E{"xyz"[c]}|²')
        ax3.set(xlabel='x [µm]', ylabel='normalized', title='Components along x through the center of the XY plane')
        ax3.legend(frameon=False)
        pdf.savefig(fig); plt.close(fig)


def slm_phase_image(r, path, width=1920, height=1080, radius=500, cx=0, cy=0, gray_2pi=256):
    """Write the correction phase as a grayscale SLM image (wrapped, gray_2pi levels per 2 pi), plus a CSV of the
    unwrapped phase versus pupil radius and a JSON file with the settings."""
    import json
    from pathlib import Path
    from PIL import Image
    s = r.scene
    th, ph = r.correction
    sin_max = np.sin(th[-1])
    yy, xx = np.mgrid[0:height, 0:width]
    rr = np.hypot(xx - (width-1)/2 - cx, yy - (height-1)/2 - cy)/radius
    inside = rr <= 1
    phase = np.interp(np.arcsin(np.clip(rr, 0, 1)*sin_max), th, ph)
    level = np.floor(np.mod(phase, 2*np.pi)/(2*np.pi)*gray_2pi)
    level = np.where(inside, np.clip(level, 0, gray_2pi-1), 0)
    if gray_2pi <= 256:
        Image.fromarray(level.astype(np.uint8), 'L').save(path)
    else:
        Image.fromarray(level.astype(np.uint16)).save(path)
    base = Path(path).with_suffix('')
    rho = s.focal_length*np.sin(th)
    np.savetxt(str(base)+'.csv', np.c_[rho/1000, np.degrees(th), ph, radius*np.sin(th)/sin_max], delimiter=',',
               header='pupil_radius_mm,theta_deg,phase_rad,slm_radius_px', comments='')
    meta = dict(image=Path(path).name, width_px=width, height_px=height, pupil_radius_px=radius, centre_offset_px=[cx, cy],
                grey_levels_per_2pi=gray_2pi, target_z_um=s.correct_z, wavelength_um=s.wavelength, na=s.na,
                focal_length_um=s.focal_length, immersion_index=s.layers[0].n_re,
                stack=[dict(name=l.name, n=[l.n_re, l.n_im], d_um=l.d) for l in s.layers], z_out_um=s.z_out,
                convention='Multiply the pupil field by exp(+i phase). The phase is radially symmetric, so image '
                           'orientation does not matter; check the sign convention of the SLM (invert the image if '
                           'the device applies exp(-i phase)).')
    Path(str(base)+'.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    return path
