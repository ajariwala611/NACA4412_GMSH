import gmsh, numpy as np, math, sys
import os

gmsh.initialize()
gmsh.model.add("Cube_Ygraded")

# --- Parameters ---
L = 1.0       # Cube size
Nz = 20       # Base resolution in Z (gets refined by field)
Nx = 20       # Base resolution in X
Ny_layers = 20  # Layers along Y
y_span = L     # Total height in Y

# --- Create base surface at Y = 0 ---
p0 = gmsh.model.geo.addPoint(0, 0, 0)
p1 = gmsh.model.geo.addPoint(L, 0, 0)
p2 = gmsh.model.geo.addPoint(L, 0, L)
p3 = gmsh.model.geo.addPoint(0, 0, L)

l1 = gmsh.model.geo.addLine(p0, p1)
l2 = gmsh.model.geo.addLine(p1, p2)
l3 = gmsh.model.geo.addLine(p2, p3)
l4 = gmsh.model.geo.addLine(p3, p0)

loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
surf = gmsh.model.geo.addPlaneSurface([loop])

# --- Transfinite to make it structured ---
gmsh.model.geo.mesh.setTransfiniteCurve(l1, Nx)
gmsh.model.geo.mesh.setTransfiniteCurve(l3, Nx)
gmsh.model.geo.mesh.setTransfiniteCurve(l2, Nz)
gmsh.model.geo.mesh.setTransfiniteCurve(l4, Nz)
gmsh.model.geo.mesh.setTransfiniteSurface(surf)
gmsh.model.geo.mesh.setRecombine(2, surf)

gmsh.model.geo.synchronize()

# --- Extrude along Y with recombination ---
ext = gmsh.model.geo.extrude([(2, surf)], 0, y_span, 0,
                             numElements=[Ny_layers],
                             recombine=True)

gmsh.model.geo.synchronize()

# --- Add MathEval field to refine near Y = 0 ---
gmsh.model.mesh.field.add("MathEval", 1)
gmsh.model.mesh.field.setString(1, "F", "0.005 + 0.05 * abs(y - 0.5)")  # Peak refinement at y=0.5
gmsh.model.mesh.field.setAsBackgroundMesh(1)

# --- Mesh Options ---
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.Recombine3DAll", 1)
gmsh.option.setNumber("Mesh.Recombine3DLevel", 0)

# --- Generate and Write ---
gmsh.model.mesh.generate(3)
script_dir = os.path.dirname(os.path.abspath(__file__))
# gmsh.write(os.path.join(script_dir, "graded_cube.msh"))
# gmsh.write(os.path.join(script_dir, "graded_cube.vtk"))

if "-nopopup" not in sys.argv:
    gmsh.fltk.run()
gmsh.finalize()
