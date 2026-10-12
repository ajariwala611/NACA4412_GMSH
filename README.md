# NACA airfoil meshing with Gmsh

This repository currently contains a NACA four-digit generator for a structured 3D
external-flow mesh with nested blocks across the span, and a Gmsh-to-Exodus
converter for use with exo2nek.

The nested generator accepts an airfoil code, JSON configuration, and command-line
overrides. Planar 2D and simple spanwise extrusion remain planned modes.

## Current files

| Path | Purpose |
| --- | --- |
| `nested_z_3D_blocks_geo_normal.py` | Active nested 3D generator with surface-normal offsets, wake grading, physical groups, and mesh-quality reporting. |
| `gmsh_to_exo_exo2nek.py` | Converts 3D Gmsh volume meshes to Exodus and reconstructs boundary side sets for exo2nek. |
| `naca_mesh/` | Shared airfoil geometry, configuration, grading, and command-line controls. |
| `examples/nested_coarse.json` | Small nested-grid configuration for quick checks; JSON counts are legacy node counts. |
| `tests/` | Numerical checks and small end-to-end meshes with conformity, boundary, and quality checks. |
| `legacy/` | Older generators and technique experiments, retained as references. See [legacy/README.md](legacy/README.md). |

## Running the current workflow

The generator requires Python, `gmsh`, and `numpy`. The converter additionally
requires `meshio` and `netCDF4`. Validated versions are pinned in `requirements.txt`.
Install them into your chosen Python environment with `python -m pip install -r requirements.txt`.

Run with the preserved default parameters:

```sh
python nested_z_3D_blocks_geo_normal.py -nopopup
python gmsh_to_exo_exo2nek.py nested_z_3D_blocks_geo.msh nested_z_3D_blocks_geo.exo
```

Without `-nopopup`, the generator opens the Gmsh GUI. It currently writes mesh
files and a quality report beside the script:

- `nested_z_3D_blocks_geo.msh`
- `nested_z_3D_blocks_geo.cgns`
- `nested_z_3D_blocks_geo_quality_order2.txt`

The converter creates an Exodus file; running exo2nek is a separate step.

For a quick coarse mesh with first-order elements and a separate output folder:

```sh
python nested_z_3D_blocks_geo_normal.py --no-gui \
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
`--domain-y` sets the outer Y extent. All current topology lengths use a unit
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
from nested_z_3D_blocks_geo_normal import generate_nested

paths = generate_nested(Params(airfoil="4412"), output_dir="meshes", gui=False)
```

Importing the generator creates no mesh. `generate_nested` owns and finalizes
its Gmsh session, including when geometry generation or export raises an error.

## Planned mesh modes

| Mode | Intended behavior | Status |
| --- | --- | --- |
| `2d` | Build a structured planar external-flow grid, with explicit element counts over the nose, upper surface, and lower surface. | Planned |
| `3d-extrude` | Extrude the same planar block grid along Z to create hexahedra. | Planned |
| `3d-nested` | Build conforming blocks with nested resolution across the span. This will remain the default mode. | Implemented as a callable generator with four-digit geometry and configuration; unified mode selection planned. |

All modes should share the airfoil definition, domain dimensions, wall-normal
grading, nose split, and wake controls. Span and spanwise resolution should
apply only to the 3D modes. The nested mode may require additional topology
constraints and parameters beyond those used for simple extrusion.

Changing the current 3D generator to `mesh.generate(2)` is not a planar-grid
implementation: it would mesh surfaces of the 3D geometry.

### Planned grid images

Each mesh mode will save PNG previews in an `images/` subfolder beside its mesh
output, with filenames identifying the airfoil and mode:

- **2D:** full domain and airfoil close-up.
- **3D extrusion:** perspective view and spanwise cross-section.
- **3D nested:** perspective view and cross-sections showing the spanwise blocks.

PNG export is a planned feature and is not yet implemented.

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
3. Build a shared planar block topology with the intended nose, upper, lower,
   wall-normal, and wake controls. Verify a 2D mesh before adding extrusion.
4. Add simple Z extrusion of that planar topology, preserving conformity and
   propagating physical boundary groups.
5. Integrate the existing nested 3D construction behind the same configuration
   and mode-selection interface. Check block interfaces and mesh quality.
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
