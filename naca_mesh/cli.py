"""Command-line configuration for the current nested generator."""

import argparse
import json
from pathlib import Path

from .config import Params


def main(generator=None):
    parser = argparse.ArgumentParser(description="Generate a NACA four-digit XY grid or 3D mesh.")
    parser.add_argument("--mode", choices=("2d", "3d-extrude", "3d-nested"), default="3d-nested")
    parser.add_argument("--z-elements", type=int, help="Uniform spanwise elements for 3d-extrude (default: 25)")
    parser.add_argument("--z-mid-elements", type=int, help="Elements in the middle Z segment for 3d-nested")
    parser.add_argument("--z-side-elements", type=int, help="Elements in each side Z segment for 3d-nested")
    parser.add_argument("--wall-elements", type=int, help="Elements across the inner wall-normal blocks")
    parser.add_argument("--outer-elements", type=int, help="Elements across the outer wall-normal blocks")
    parser.add_argument("--config", type=Path, help="JSON object with Params fields; counts in JSON are legacy node counts")
    parser.add_argument("--airfoil", help="Four-digit code (default: 4412)")
    parser.add_argument("--upper-elements", type=int, help="Elements along the upper airfoil segment")
    parser.add_argument("--lower-elements", type=int, help="Elements along the lower airfoil segment")
    parser.add_argument("--nose-elements", type=int, help="Elements along the nose segment")
    parser.add_argument("--span", type=float, help="Total Z span")
    parser.add_argument("--domain-x", type=float, help="Outlet X coordinate")
    parser.add_argument("--domain-y", type=float, help="Outer Y extent (top/bottom at +/- this value)")
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: generator directory)")
    parser.add_argument("--prefix", help="Output filename prefix (default: airfoil and mode)")
    parser.add_argument("--order", type=int, choices=(1, 2), default=2)
    parser.add_argument("-nopopup", "--no-gui", action="store_true", help="Run without opening the Gmsh GUI")
    args = parser.parse_args()
    if args.z_elements is not None and args.mode != "3d-extrude":
        parser.error("--z-elements applies only to --mode 3d-extrude")
    if (args.z_mid_elements is not None or args.z_side_elements is not None) and args.mode != "3d-nested":
        parser.error("--z-mid-elements and --z-side-elements apply only to --mode 3d-nested")
    values = {}
    if args.config:
        try:
            values = json.loads(args.config.read_text())
            if not isinstance(values, dict):
                raise ValueError("configuration must be a JSON object")
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    for flag, field in (("airfoil", "airfoil"), ("span", "W"), ("z_elements", "z_elements"),
                        ("domain_x", "Lx"), ("domain_y", "Ly")):
        if getattr(args, flag) is not None:
            values[field] = getattr(args, flag)
    for flag, field in (("upper_elements", "upper_surface"),
                        ("lower_elements", "lower_surface"), ("nose_elements", "nose_surface"),
                        ("wall_elements", "ny"), ("outer_elements", "Ny_outer"),
                        ("z_mid_elements", "nz_mid"), ("z_side_elements", "nz_left_right")):
        count = getattr(args, flag)
        if count is not None:
            if count < 1:
                parser.error(f"--{flag.replace('_', '-')} must be >= 1")
            values[field] = count + 1
    try:
        params = Params(**values)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if generator is None:
        from generate_mesh import generate_mesh
        generator = generate_mesh
    paths = generator(params, mode=args.mode, output_dir=args.output_dir, prefix=args.prefix,
                      order=args.order, gui=not args.no_gui)
    for kind, path in paths.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
