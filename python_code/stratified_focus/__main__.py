"""Headless sample run: python -m stratified_focus --output focus_results.npz."""
import argparse
from . import SimulationConfig, Sampling, simulate
from .session import save_result


def main():
    parser = argparse.ArgumentParser(description='Run a sampled vectorial focusing simulation')
    parser.add_argument('--output', default='focus_results.npz', help='Compressed NumPy output')
    parser.add_argument('--xy', type=int, default=81, help='XY samples per dimension')
    parser.add_argument('--theta', type=int, default=48)
    parser.add_argument('--phi', type=int, default=64)
    parser.add_argument('--axial', action='store_true', help='Also compute XZ and YZ')
    parser.add_argument('--volume', action='store_true', help='Also compute the raw XYZ volume on the CPU')
    parser.add_argument('--volume-xy', type=int, default=41, help='Volume samples per lateral axis')
    parser.add_argument('--volume-z', type=int, default=41, help='Volume axial samples')
    args = parser.parse_args()
    config = SimulationConfig(sampling=Sampling(phi=args.phi, theta=args.theta, xy=args.xy,
        compute_axial=args.axial, compute_volume=args.volume, volume_xy=args.volume_xy, volume_z=args.volume_z))
    result = simulate(config)
    save_result(args.output, result)
    print(f'Saved {args.output}; compute time = {result.elapsed_seconds:.3f} s; normalization = {result.normalization:.6g}')


if __name__ == '__main__': main()
