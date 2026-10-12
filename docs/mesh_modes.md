# Shared XY blocks and the three mesh modes

## Geometry organization

`naca_mesh/geometry.py` owns the three airfoil splines and the analytic-normal
offset nose spline. The direct XY builder and original nested builder use these
same helpers and the shared four-digit coordinates in `naca_mesh/airfoil.py`.

`naca_mesh/planar.py` directly creates the full external-flow domain at Z=0.
Its 14 quadrilateral blocks are:

- Inner nose, upper airfoil, and lower airfoil: three blocks.
- Upper/lower near wake and upper/lower far wake: four blocks.
- Outer nose, upper outer domain, and lower outer domain: three blocks.
- Upper/lower outer near wake and upper/lower outer far wake: four blocks.

The inner nose boundary is the existing normal-offset spline, and the outer
inlet is the existing semicircle. Domain dimensions, grading, wake stretching,
and surface counts follow the original XY layout. Each curve is created once
and reused by adjoining blocks. No YZ sections or volumes are constructed.

`naca_mesh/extruded.py` extrudes all 14 planar surfaces together along Z with
`numElements=[z_elements]` and recombination. N requested elements create N
uniform hexahedral layers and N+1 planes of corner nodes. Quadratic elements
also have intermediate nodes; these do not change the layer count.

`generate_mesh.py:build_nested_geometry` retains the original nested geometry
sequence independently of the simple extrusion path. `generate_mesh` selects
the builder, generates the appropriate mesh dimension, and writes outputs.
`generate_nested` is a compatibility wrapper with the earlier output prefix.

## Boundaries and counts

2D meshes contain physical curves named Airfoil, Inlet, Outlet, Top, and Bottom,
and a physical surface group named fluid. Extrusion carries these names onto
the swept surfaces and adds Side1/Side2 caps and the fluid volume group. The
existing nested labels are preserved.

The surface controls specify elements on the upper, nose, and lower splines.
Opposing block curves use matching transfinite divisions. Wall-normal grading
and resolution and near/far wake counts retain configurable defaults.

## Validation

The direct coarse NACA 4412 XY grid was compared with the original nested
generator's Z=0 grid. Both contain 1,019 exported nodes and 930 quads. Maximum
nearest-node distance was approximately 1.81e-10 chord lengths. Element tags and
ordering differ because the direct builder creates entities in a simpler order.

Seven Z layers produce exactly 930 x 7 = 6,510 hexahedra. The first-order minimum
sampled scaled Jacobian was approximately 0.294854289 in both the XY and extruded
grids, with no nonpositive values.

Automated mode tests cover 4412 and 0012, first- and second-order elements, one
and seven extrusion layers, zero 2D Z coordinates, exact surface element counts,
mesh conformity, and complete non-overlapping boundary labels. They also prevent
planar modes from calling the nested geometry builder.

The full default quadratic nested `.msh` and quality report were regenerated
after separating the builders and match the original baseline byte for byte.
An added synchronization after duplicate removal changed the nested mesh; that
extra operation was removed to retain the original sequence.

The coarse first-order extrusion converted to Exodus with all seven boundary
side sets. exo2nek itself was not run. The converter handles 3D volume meshes,
not 2D grids.

## Remaining work

Automatic PNG previews are now implemented in `naca_mesh/previews.py`; see
[the example gallery](../examples/README.md). Further work can separate mode-specific
configuration validation, support non-unit chords/open trailing edges, and
optimize measured costs. Each new airfoil and parameter combination still needs
quality checks; these results do not prove validity for all four-digit profiles.
