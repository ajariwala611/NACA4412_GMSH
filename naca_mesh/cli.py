"""Command-line configuration for the current nested generator."""

import argparse
import json
from pathlib import Path

from .config import Params


def main(generator=None):
    parser = argparse.ArgumentParser(description="Generate a NACA four-digit nested spanwise mesh.")
    parser.add_argument("--config", type=Path, help="JSON object with Params fields; counts in JSON are legacy node counts")
    parser.add_argument("--airfoil", help="Four-digit code (default: 4412)")
    parser.add_argument("--upper-elements", type=int, help="Elements along the upper airfoil segment")
    parser.add_argument("--lower-elements", type=int, help="Elements along the lower airfoil segment")
    parser.add_argument("--nose-elements", type=int, help="Elements along the nose segment")
    parser.add_argument("--span", type=float, help="Total Z span")
    parser.add_argument("--domain-x", type=float, help="Outlet X coordinate")
    parser.add_argument("--domain-y", type=float, help="Outer Y extent (top/bottom at +/- this value)")
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: generator directory)")
    parser.add_argument("--prefix", default="nested_z_3D_blocks_geo", help="Output filename prefix")
    parser.add_argument("--order", type=int, choices=(1, 2), default=2)
    parser.add_argument("-nopopup", "--no-gui", action="store_true", help="Run without opening the Gmsh GUI")
    args = parser.parse_args()
    values = {}
    if args.config:
        try:
            values = json.loads(args.config.read_text())
            if not isinstance(values, dict):
                raise ValueError("configuration must be a JSON object")
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    for flag, field in (("airfoil", "airfoil"), ("span", "W"),
                        ("domain_x", "Lx"), ("domain_y", "Ly")):
        if getattr(args, flag) is not None:
            values[field] = getattr(args, flag)
    for flag, field in (("upper_elements", "upper_surface"),
                        ("lower_elements", "lower_surface"), ("nose_elements", "nose_surface")):
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
        from nested_z_3D_blocks_geo_normal import generate_nested
        generator = generate_nested
    paths = generator(params, output_dir=args.output_dir, prefix=args.prefix,
                      order=args.order, gui=not args.no_gui)
    for kind, path in paths.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
