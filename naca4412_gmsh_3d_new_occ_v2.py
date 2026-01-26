import gmsh
import numpy as np
import math
import os

# Parameters
c = 1.0
R = 5.0 * c
Lwake = 10.0 * c
Nu, Nl = 100, 50
Ntheta = 10
lc_wall = 5e-3 * c
lc_far = 5.0 * c
Lz = 0.1 * c
Nz_outer = 3
Nz_inner = 6
script_dir = os.path.dirname(os.path.abspath(__file__))

def cosine_spacing(N, start=0.0, end=1.0):
    beta = np.linspace(0, math.pi, N)
    x = 0.5 * (1 - np.cos(beta))
    return start + (end - start) * x

def naca4412(x, closed_TE=True):
    m, p, t = 0.04, 0.4, 0.12
    yt = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - (0.1036 if closed_TE else 0.1015)*x**4)
    yc  = np.where(x < p, m/p**2*(2*p*x - x**2), m/(1-p)**2*((1-2*p)+2*p*x - x**2))
    dyc = np.where(x < p, 2*m/p**2*(p-x), 2*m/(1-p)**2*(p-x))
    th  = np.arctan(dyc)
    xu, yu = x - yt*np.sin(th), yc + yt*np.cos(th)
    xl, yl = x + yt*np.sin(th), yc - yt*np.cos(th)
    return xu, yu, xl, yl

# Initialize GMSH
gmsh.initialize()
gmsh.model.add("NACA4412_CGrid_OCC")

# Airfoil wire
xU_cos = cosine_spacing(Nu + 1, 1.0, 0.0)
xL_cos = cosine_spacing(Nl + 1, 0.0, 1.0)
xu, yu, _, _ = naca4412(xU_cos)
_, _, xl, yl = naca4412(xL_cos)

pt_TE = gmsh.model.occ.addPoint(c, 0.0, 0.0, lc_wall)
pt_LE = gmsh.model.occ.addPoint(0.0, 0.0, 0.0, lc_wall)

upper_pts = [pt_TE] + [gmsh.model.occ.addPoint(x * c, y * c, 0, lc_wall) for x, y in zip(xu[1:-1], yu[1:-1])] + [pt_LE]
lower_pts = [pt_LE] + [gmsh.model.occ.addPoint(x * c, y * c, 0, lc_wall) for x, y in zip(xl[1:-1], yl[1:-1])] + [pt_TE]

curve_upper = gmsh.model.occ.addSpline(upper_pts)
curve_lower = gmsh.model.occ.addSpline(lower_pts)
airfoil_wire = gmsh.model.occ.addWire([curve_upper, curve_lower], checkClosed=True)
gmsh.model.occ.synchronize()
# Outer domain
theta = np.linspace(math.pi/2, 3*math.pi/2, Ntheta)
semi = [gmsh.model.occ.addPoint(R*math.cos(t), R*math.sin(t), 0, lc_far) for t in theta]
p_bot = gmsh.model.occ.addPoint(c + Lwake, -R, 0, lc_far)
p_top = gmsh.model.occ.addPoint(c + Lwake,  R, 0, lc_far)

lines = [gmsh.model.occ.addLine(semi[i], semi[i+1]) for i in range(len(semi)-1)]
lines += [
    gmsh.model.occ.addLine(semi[-1], p_bot),
    gmsh.model.occ.addLine(p_bot, p_top),
    gmsh.model.occ.addLine(p_top, semi[0])
]
outer_wire = gmsh.model.occ.addWire(lines, checkClosed=True)
gmsh.model.occ.synchronize()
# Surfaces
outer_surf = gmsh.model.occ.addPlaneSurface([outer_wire])
# airfoil_surf = gmsh.model.occ.addPlaneSurface([airfoil_wire])
# Duplicate curves for airfoil volume
curve_upper_copy = gmsh.model.occ.copy([(1, curve_upper)])[0][1]
curve_lower_copy = gmsh.model.occ.copy([(1, curve_lower)])[0][1]
airfoil_wire_copy = gmsh.model.occ.addWire([curve_upper_copy, curve_lower_copy], checkClosed=True)
airfoil_surf = gmsh.model.occ.addPlaneSurface([airfoil_wire_copy])

gmsh.model.occ.synchronize()
# Extrude
outer_vol = gmsh.model.occ.extrude([(2, outer_surf)], 0, 0, Lz,numElements=[Nz_outer],recombine=True)
# print("Outer volume:", outer_vol)
airfoil_vol = gmsh.model.occ.extrude([(2, airfoil_surf)], 0, 0, Lz,numElements=[Nz_inner],recombine=True)
# print("Airfoil volume:", airfoil_vol)
gmsh.model.occ.synchronize()

vol_outer = [e for e in outer_vol if e[0] == 3][0]
# print("Outer volume:", vol_outer)
vol_airfoil = [e for e in airfoil_vol if e[0] == 3][0]
# print("Airfoil volume:", vol_airfoil)
# Cut airfoil out
# cut_vol,_ = gmsh.model.occ.cut([vol_outer],[vol_airfoil])
gmsh.model.occ.cut([vol_outer],[vol_airfoil])
gmsh.model.occ.synchronize()

# print("Cut volume:", cut_vol)

# airfoil_bdry_pts = gmsh.model.getBoundary([vol_airfoil],recursive=True)
# print("Boundary entities:", airfoil_bdry_pts)

# outer_bdry_pts = gmsh.model.getBoundary([cut_vol[0]],recursive=True)
# print("outer entities:", outer_bdry_pts)
# entity = gmsh.model.getEntities(0)
# print("Entities:", entity)
# gmsh.model.mesh.setSize(airfoil_bdry_pts,0.01)
# gmsh.model.mesh.setSize(outer_bdry_pts,0.025)

# gmsh.model.mesh.field.add("Box", 1)
# gmsh.model.mesh.field.setNumber(1, "VIn", 0.01)
# gmsh.model.mesh.field.setNumber(1, "VOut", 0.1)
# gmsh.model.mesh.field.setNumber(1, "XMin", -0.1)
# gmsh.model.mesh.field.setNumber(1, "XMax", 1.1)
# gmsh.model.mesh.field.setNumber(1, "YMin", -0.2)
# gmsh.model.mesh.field.setNumber(1, "YMax", 0.2)
# gmsh.model.mesh.field.setNumber(1, "ZMin", 0.0)
# gmsh.model.mesh.field.setNumber(1, "ZMax", 0.1)
# gmsh.model.mesh.field.setNumber(1, "Thickness", 0.1)
# Let's use the minimum of all the fields as the mesh size field:
# gmsh.model.mesh.field.add("Min", 2)
# gmsh.model.mesh.field.setNumbers(2, "FieldsList", [1])
# gmsh.model.mesh.field.setAsBackgroundMesh(2)


# Mesh

gmsh.option.setNumber("Mesh.Algorithm",     5)  # quad-delaunay (fast)
gmsh.option.setNumber("Mesh.Algorithm3D",      1)  
gmsh.option.setNumber("Mesh.RecombineAll",   1)  # recombine tri → quad
gmsh.option.setNumber("Mesh.RecombinationAlgorithm",   3)  #
gmsh.option.setNumber("Mesh.Recombine3DAll",   1)  # recombine tetra → hex
gmsh.option.setNumber("Mesh.Recombine3DLevel",   0)  #
# gmsh.option.setNumber("Mesh.Recombine3DConformity",   0)  #
# gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)  # simple subdivision

gmsh.write(os.path.join(script_dir, "debug_output.geo_unrolled"))
# exit()

gmsh.model.mesh.generate(3)

# Write
gmsh.write(os.path.join(script_dir, "naca4412_cgrid3D_bool.msh"))
gmsh.write(os.path.join(script_dir, "naca4412_cgrid3D_bool.cgns"))
gmsh.finalize()
