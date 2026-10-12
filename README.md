# NACA airfoil meshing with Gmsh

This repository generates structured external-flow grids around NACA four-digit
airfoils: a full planar XY grid, its uniform-Z extrusion, or the original nested
spanwise 3D grid. It also includes a Gmsh-to-Exodus converter for exo2nek.

The generator accepts an airfoil code, JSON configuration, and command-line
overrides. Nested 3D remains the default mode.

## Current files

| Path | Purpose |
| --- | --- |
| `generate_mesh.py` | Mode selection and original nested 3D geometry assembly, with physical groups and quality reporting. |
| `gmsh_to_exo_exo2nek.py` | Converts 3D Gmsh volume meshes to Exodus and reconstructs boundary side sets for exo2nek. |
| `naca_mesh/` | Shared spline geometry, direct XY block builder, simple extrusion, configuration, grading, and command-line controls. |
| `examples/nested_coarse.json` | Small nested-grid configuration for quick checks; JSON counts are legacy node counts. |
| `tests/` | Numerical checks and small end-to-end meshes with conformity, boundary, and quality checks. |
| `legacy/` | Older generators and technique experiments, retained as references. See [legacy/README.md](legacy/README.md). |

## Running the current workflow

The generator requires Python, `gmsh`, `numpy`, and `matplotlib` for PNG export. The converter additionally
requires `meshio` and `netCDF4`. Validated versions are pinned in `requirements.txt`.
Install them into your chosen Python environment with `python -m pip install -r requirements.txt`.

Run with the preserved default parameters:

```sh
python generate_mesh.py -nopopup
python gmsh_to_exo_exo2nek.py naca4412_3d-nested.msh naca4412_3d-nested.exo
```

Without `-nopopup`, the generator opens the Gmsh GUI. It currently writes mesh
files and a quality report beside the script:

- `naca4412_3d-nested.msh`
- `naca4412_3d-nested.cgns`
- `naca4412_3d-nested_quality_order2.txt`

The converter creates an Exodus file; running exo2nek is a separate step.

For a quick coarse mesh with first-order elements and a separate output folder:

```sh
python generate_mesh.py --no-gui \
  --config examples/nested_coarse.json --airfoil 0012 \
  --upper-elements 30 --lower-elements 15 --nose-elements 20 \
  --order 1 --output-dir meshes --prefix naca0012
```

Use `--help` for available overrides. Command-line surface counts specify
**elements**, while `upper_surface`, `lower_surface`, and `nose_surface` in JSON
retain their original **node-count** convention. The default segment element
counts are therefore 274 upper, 49 lower, and 149 nose. Command-line overrides
take precedence over JSON. A full-resolution JSON can omit any default field;
default values live in `naca_mesh/config.py`.

`--span` sets the total Z extent, `--domain-x` sets the outlet X coordinate, and
`--domain-y` sets the outer Y extent. `--wall-elements` and `--outer-elements`
control the inner and outer wall-normal block resolution. All current topology lengths use a unit
chord. Non-unit chord and open trailing edges are rejected explicitly until
the geometry/topology has been adapted for them. The existing nose offset also
requires `0.05 < h_in < H < Ly`. Invalid configurations are checked before
Gmsh starts, but mesh quality still needs checking for each new parameter set.

The geometry definition supports four-digit codes, including symmetric
profiles. Small meshes have been checked for 4412, 0012, 2412, and 4415; this
does not establish mesh quality for every airfoil or parameter combination.
The wall boundary is identified by its constructed patches, rather than a
fixed airfoil bounding box.

For Python callers:

```python
from naca_mesh import Params
from generate_mesh import generate_mesh

paths = generate_mesh(Params(airfoil="4412", z_elements=32), mode="3d-extrude",
                      output_dir="meshes", gui=False)
```

Importing the generator creates no mesh. `generate_mesh` owns and finalizes
its Gmsh session, including when geometry generation or export raises an error.
`generate_nested` remains available as a compatibility entry point, retaining
the earlier `nested_z_3D_blocks_geo` output prefix.

## Mesh modes

| Mode | Intended behavior | Status |
| --- | --- | --- |
| `2d` | Build a full external-flow grid in XY at Z=0, with nose/upper/lower surface element controls. | Implemented directly as 14 planar blocks; no volumes are constructed. |
| `3d-extrude` | Extrude that same XY block grid along Z into uniform hexahedral layers. | Implemented; `--z-elements N` creates N layers (N+1 corner-node planes). |
| `3d-nested` | Use the original conforming nested spanwise block construction. | Implemented and remains the default. |

```sh
# Full XY grid, using a small configuration for a quick run
python generate_mesh.py --mode 2d --config examples/nested_coarse.json --no-gui

# Same XY grid, with 32 elements along the span
python generate_mesh.py --mode 3d-extrude --z-elements 32 --span 0.1 \
  --config examples/nested_coarse.json --no-gui

# Original nested topology
python generate_mesh.py --mode 3d-nested --config examples/nested_coarse.json --no-gui
```

The 2D builder uses three airfoil splines, the normal-offset inner nose spline,
the outer inlet semicircle, upper/lower inner and outer blocks, and near/far wake
blocks. Simple extrusion uses the same planar surfaces in a single shared
extrusion operation. The nested builder retains the original geometry sequence.
See [docs/mesh_modes.md](docs/mesh_modes.md) for the organization and checks.

All modes share the airfoil definition, domain dimensions, wall-normal
grading, nose split, and wake controls. Span and spanwise resolution should
apply only to the 3D modes. For nested mode, `--z-mid-elements` and
`--z-side-elements` override the original middle and side Z-segment node counts.
`--z-elements` applies only to simple extrusion. The current shared configuration
still validates legacy nested fields even for 2D; unused values can be left at
their defaults.

Changing the current 3D generator to `mesh.generate(2)` is not a planar-grid
implementation: it would mesh surfaces of the 3D geometry.

### Grid images

Each mesh mode saves PNG previews in an `images/` subfolder beside its mesh
output, with filenames identifying the airfoil and mode:

- **2D:** full domain and airfoil close-up.
- **3D extrusion:** XY domain/close-up, airfoil perspective, and YZ block section.
- **3D nested:** XY domain/close-up, airfoil perspective, and YZ section showing the spanwise blocks.

Rendering uses the actual generated surface mesh and works without the Gmsh
GUI. Quadratic edges include their middle nodes. The 3D display aspect expands
the span for readability, as labeled on the figure; coordinates are unchanged.
Use `--no-images` to skip rendering, `--image-dir PATH` to choose a PNG folder,
and `--image-dpi N` to set resolution (default: 160). Python callers can pass
`images=False`, `image_dir=...`, or `image_dpi=...` to the generator.

See [examples/README.md](examples/README.md) for the three configurations,
committed PNGs, and regeneration commands. Configurations may include `mode`
and `order`; command-line overrides take precedence.

## Parameter conventions for the refactor

- Accept a four-digit airfoil code, such as `4412` or `0012`, and derive camber,
  camber position, and thickness in one shared geometry module. Handle symmetric
  airfoils explicitly to avoid division by zero when the camber position is zero.
- Expose **element counts** to the user. For an open transfinite curve, pass
  `elements + 1` as the node count to Gmsh. Existing parameters such as
  `upper_surface`, `lower_surface`, and `nose_surface` currently specify nodes;
  preserve their effective resolution when migrating defaults.
- Keep geometry sampling counts separate from mesh element counts.
- Document lengths and scaling consistently, including chord, domain bounds,
  span, wall offsets, and wake lengths. Preserve existing geometry while making
  the units explicit.
- Keep wall-normal and wake resolution configurable; surface counts and domain
  dimensions alone do not fully determine a structured grid.
- Give each mode an explicit output path and meaningful physical boundary names.
  Planar meshes require physical curves; volume meshes require physical surfaces.

## Implementation sequence

1. **Completed:** establish a full-resolution NACA 4412 baseline and compare
   the refactored nested generator. See [docs/refactor_validation.md](docs/refactor_validation.md).
2. **Completed:** extract configuration, four-digit airfoil geometry, and grading
   utilities, remove duplicate hard-coded airfoil definitions, and introduce a
   callable entry point and command-line controls.
3. **Completed:** build the direct planar XY block topology with nose, upper,
   lower, wall-normal, and wake controls; verify first- and second-order meshes.
4. **Completed:** add simple Z extrusion, preserving conformity and physical
   boundary groups, with explicit spanwise element counts.
5. **Completed:** integrate the existing nested construction behind the shared
   configuration and mode interface; retain its original geometry sequence.
6. Add example configurations, a command-line interface, dependency metadata,
   and documented output/conversion steps. Optimize measured bottlenecks after
   the meshes are validated.

A possible target layout for the remaining work is:

```text
naca_mesh/
    config.py
    airfoil.py
    grading.py
    planar.py
    extruded.py
    nested.py
    boundaries.py
    quality.py
    cli.py
tools/
    gmsh_to_exo_exo2nek.py
examples/
tests/
legacy/
README.md
```

Validation should cover NACA 4412 and a symmetric four-digit airfoil, requested
surface element counts, conforming interfaces, expected quad/hex element types,
positive element Jacobians, and complete boundary groups. Check Exodus side
sets for the supported 3D workflow; the current converter is not a 2D exporter.

Run the current checks with:

```sh
python -m unittest discover -s tests -v
```

Gmsh integration tests skip when the Gmsh package is unavailable. Numerical
tests still require NumPy. Generated mesh files, reports, Python caches, and
macOS metadata are excluded from Git.
