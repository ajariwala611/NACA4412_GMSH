# Nested-generator refactor validation

The reference was a copy of the working `nested_z_3D_blocks_geo_normal.py`
before this refactor, including the user's existing uncommitted geometry and
quality-report changes. It was not the older committed version.

## Full-resolution default NACA 4412

Both the original and refactored scripts were run headlessly with their default
mesh settings and quadratic elements. The final refactor preserves the geometry
operation order and the original floating-point arithmetic for the analytic
normals.

| Check | Original | Refactored |
| --- | ---: | ---: |
| Total Gmsh nodes, including entities not exported | 3,429,828 | 3,429,828 |
| Total Gmsh elements, including entities not exported | 544,869 | 544,869 |
| Exported `.msh` nodes | 3,424,706 | 3,424,706 |
| Exported `.msh` elements across all dimensions | 479,294 | 479,294 |
| Quadratic hexahedra (27-node) | 410,820 | 410,820 |
| Minimum sampled scaled Jacobian (`minSJ`) | 0.740080592 | 0.740080592 |
| Elements with sampled `minSJ <= 0` | 0 | 0 |

Streaming comparison of the exported meshes found zero coordinate difference
and zero differing element records. Physical-name records and node/element IDs
also matched. The quality reports matched exactly. This establishes preservation
of the tested default mesh, rather than merely successful mesh generation.

## Numerical checks

The original and extracted default calculations produced identical sampled
upper/lower coordinates, outer grading arrays, and analytic normals. Unit tests
also cover four-digit code validation, symmetric-profile geometry without
division by zero, scalar coordinates, closed trailing edges, unit outward
normals perpendicular to the surfaces, configuration overrides, invalid nested
parameters, and uniform geometric grading.

## Coarse nested meshes

End-to-end tests generate first-order NACA 4412, 0012, and 4415 meshes using
the settings in `examples/nested_coarse.json`. They verify:

- Volume elements are hexahedra and sampled scaled Jacobians are positive.
- Hex faces have incidence one or two; the incidence-one faces exactly match
  the labeled boundary quad faces.
- Every exterior face has exactly one physical boundary label.
- All seven boundary groups exist and the airfoil wall contains nine patches.
- Mesh and report files are written and the Gmsh session is finalized.

A separate coarse NACA 2412 mesh also generated successfully and had complete
physical surface coverage. It is not part of the automated integration suite.

The thicker NACA 4415 check exposed a legacy boundary-labeling bug: the fixed
airfoil bounding box omitted three upper wall patches. The generator now uses
the constructed wall patches directly, preserving the NACA 4412 default labels
while covering the thicker profile.

## Conversion check

The unchanged converter successfully converted the coarse, first-order NACA
4415 mesh to Exodus. Inspection confirmed:

- NetCDF format: `NETCDF3_64BIT_OFFSET`.
- Volume elements: 5,208.
- Continuous side-set IDs: 1 through 7.
- Names: Inlet, Outlet, Airfoil, Bottom, Top, Side1, Side2.
- Total boundary sides: 2,678.

exo2nek itself was not run. The full-resolution quadratic mesh was not converted
as part of this check.

## Reproduction and limits

```sh
python -m unittest discover -s tests -v
python nested_z_3D_blocks_geo_normal.py --no-gui
```

The tests currently contain 12 test methods, including integration subcases.
Validated library versions are recorded in `requirements.txt`. Generated
comparison meshes were placed outside the repository; no multi-million-node
mesh is included in source control.

The nested topology still assumes a unit chord, a closed trailing edge, and
the original nose-offset construction. Configuration validation rejects known
unsupported settings. It does not prove validity for every four-digit profile
or resolution. New profiles and parameter combinations still require mesh
quality and boundary checks. Planar 2D and simple Z extrusion are not implemented
by this refactor.
