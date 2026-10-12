# Reference generators

These scripts preserve older approaches for comparison and selective reuse.
The active root-level generator and converter do not import them. They are not
supported modes of the planned unified generator, and their mesh quality has
not been verified during this organization change.

| Folder / file | Reason retained |
| --- | --- |
| `nested_blocks/nested_z_3D_blocks_geo.py` | Earlier nested-block generator, useful for comparing geometry and grading with the active normal-offset version. |
| `planar/naca4412_gmsh_2D_new.py` | 2D external-domain and refinement-field reference. It does not implement the planned explicit nose/upper/lower structured element-count interface. |
| `extrusion/naca4412_gmsh_3_occ_new_extrude.py` | OCC extrusion and physical-boundary reference. |
| `unstructured/naca4412_gmsh_3d_new_occ_v3.py` | Boolean-cut external domain with refinement fields and physical groups; a distinct unstructured approach. |

The scripts were moved without changing their contents. Their output paths use
their own script directory, so generated files will now appear inside the
corresponding reference folder. They keep their original filenames to make
their history and origin easy to recognize.
