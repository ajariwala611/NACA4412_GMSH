import gmsh
import math
import os
import sys
import gmsh, math, os

Lx = 1.0
Ly = 1.0
Lz = 1.0
Nx = 16
Nz = 16

BL_thickness = 0.2
N_bl         = 20
dy1          = 1e-3
Ny_outer     = 10

def solve_ratio(dy1, T, N, r_min=1.0+1e-8, r_max=50.0, tol=1e-12, max_iter=100):
    def f(r):
        return dy1 * (1.0 - r**N) / (1.0 - r) - T
    a, b = r_min, r_max
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        return 1.0
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) < tol:
            return m
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)

ratio_bl = solve_ratio(dy1, BL_thickness, N_bl)
print("r =", ratio_bl)

gmsh.initialize()
gmsh.model.add("cube_BL_y0")

# points
p0 = gmsh.model.geo.addPoint(0.0,          0.0,          0.0)
p1 = gmsh.model.geo.addPoint(Lx,           0.0,          0.0)
p2 = gmsh.model.geo.addPoint(Lx,  BL_thickness,          0.0)
p3 = gmsh.model.geo.addPoint(0.0, BL_thickness,          0.0)
p4 = gmsh.model.geo.addPoint(0.0, Ly, 0.0)
p5 = gmsh.model.geo.addPoint(Lx, Ly, 0.0)

# lower block
l0 = gmsh.model.geo.addLine(p0, p1)
l1 = gmsh.model.geo.addLine(p1, p2)   # y: 0 -> BL
l2 = gmsh.model.geo.addLine(p2, p3)
l3 = gmsh.model.geo.addLine(p3, p0)   # y: BL -> 0

loop_lower = gmsh.model.geo.addCurveLoop([l0, l1, l2, l3])
surf_lower = gmsh.model.geo.addPlaneSurface([loop_lower])

# upper block
l4 = gmsh.model.geo.addLine(p3, p4)
l5 = gmsh.model.geo.addLine(p4, p5)
l6 = gmsh.model.geo.addLine(p5, p2)

loop_upper = gmsh.model.geo.addCurveLoop([l4, l5, l6, l2])
surf_upper = gmsh.model.geo.addPlaneSurface([loop_upper])

# X-direction uniform
for line in [l0, l2, l5]:
    gmsh.model.geo.mesh.setTransfiniteCurve(line, Nx)

# Y-direction: bottom BL + outer
gmsh.model.geo.mesh.setTransfiniteCurve(l1, N_bl, coef=ratio_bl)      # fine near y=0 at p1
gmsh.model.geo.mesh.setTransfiniteCurve(l3, N_bl, coef=1.0/ratio_bl)  # fine near y=0 at p0

gmsh.model.geo.mesh.setTransfiniteCurve(l4, Ny_outer)
gmsh.model.geo.mesh.setTransfiniteCurve(l6, Ny_outer)

gmsh.model.geo.mesh.setTransfiniteSurface(surf_lower, cornerTags=[p0, p1, p2, p3])
gmsh.model.geo.mesh.setTransfiniteSurface(surf_upper, cornerTags=[p3, p2, p5, p4])
gmsh.model.geo.mesh.setRecombine(2, surf_lower)
gmsh.model.geo.mesh.setRecombine(2, surf_upper)

gmsh.model.geo.synchronize()

# extrude both surfaces to 3D
Nz = 16
gmsh.model.geo.extrude([(2, surf_lower), (2, surf_upper)],
                       0.0, 0.0, Lz,
                       numElements=[Nz],
                       recombine=True)

gmsh.model.geo.synchronize()
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.Recombine3DAll", 1)
gmsh.model.mesh.generate(3)

out = os.path.dirname(os.path.abspath(__file__))
# gmsh.write(os.path.join(out, "cube_BL_y0.msh"))

if "-nopopup" not in sys.argv:
    gmsh.fltk.run()
gmsh.finalize()
