import gmsh, numpy as np, math, os

# ---------- Parameters ----------
c = 1.0
R = 5.0 * c
Lwake = 10.0 * c
Nu, Nl = 100, 50
Ntheta = 20
lc_min = 5e-3 * c
lc_max = 4.0 * c
Lz = 0.1 * c
Nz = 10
script_dir = os.path.dirname(os.path.abspath(__file__))

# ---------- Helper Functions ----------
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

# ---------- GMSH Initialization ----------
gmsh.initialize()
gmsh.model.add("NACA4412_Field")
SetFactory = gmsh.model.occ

# ---------- Airfoil Geometry ----------
xU_cos = cosine_spacing(Nu + 1, 1.0, 0.0)
xL_cos = cosine_spacing(Nl + 1, 0.0, 1.0)
xu, yu, _, _ = naca4412(xU_cos)
_, _, xl, yl = naca4412(xL_cos)

pt_TE = SetFactory.addPoint(c, 0.0, 0.0, lc_min)
pt_LE = SetFactory.addPoint(0.0, 0.0, 0.0, lc_min)

upper_pts = [pt_TE] + [SetFactory.addPoint(x*c, y*c, 0, lc_min) for x, y in zip(xu[1:-1], yu[1:-1])] + [pt_LE]
lower_pts = [pt_LE] + [SetFactory.addPoint(x*c, y*c, 0, lc_min) for x, y in zip(xl[1:-1], yl[1:-1])] + [pt_TE]

curve_upper = SetFactory.addSpline(upper_pts)
curve_lower = SetFactory.addSpline(lower_pts)
airfoil_loop = SetFactory.addWire([curve_upper, curve_lower], checkClosed=True)

# ---------- Outer C-grid Boundary ----------
R = 5.0 * c
Lwake = 10.0 * c
theta = np.linspace(math.pi/2, 3*math.pi/2, Ntheta)
semi = [SetFactory.addPoint(R*math.cos(t), R*math.sin(t), 0, lc_max) for t in theta]
p_botOut = SetFactory.addPoint(c+Lwake, -R, 0, lc_max)
p_topOut = SetFactory.addPoint(c+Lwake,  R, 0, lc_max)

lines = [SetFactory.addLine(semi[i], semi[i+1]) for i in range(len(semi)-1)]
lines += [
    SetFactory.addLine(semi[-1], p_botOut),
    SetFactory.addLine(p_botOut, p_topOut),
    SetFactory.addLine(p_topOut, semi[0])
]
outer_loop = SetFactory.addWire(lines, checkClosed=True)

# ---------- Surface with Hole ----------
domain_surf = SetFactory.addPlaneSurface([outer_loop, airfoil_loop])
SetFactory.synchronize()

# ---------- Mesh Size Field Based on Distance to Airfoil ----------
airfoil_edges = gmsh.model.getBoundary([(1, curve_upper)], combined=False)
airfoil_edges += gmsh.model.getBoundary([(1, curve_lower)], combined=False)
edge_tags = list(set([e[1] for e in airfoil_edges if e[0] == 1]))

# gmsh.model.mesh.field.add("Distance", 1)
# gmsh.model.mesh.field.setNumbers(1, "EdgesList", edge_tags)

# gmsh.model.mesh.field.add("Threshold", 2)
# gmsh.model.mesh.field.setNumber(2, "InField", 1)
# gmsh.model.mesh.field.setNumber(2, "SizeMin", lc_min)
# gmsh.model.mesh.field.setNumber(2, "SizeMax", lc_max)
# gmsh.model.mesh.field.setNumber(2, "DistMin", 0.01 * c)
# gmsh.model.mesh.field.setNumber(2, "DistMax", 0.5 * c)

# gmsh.model.mesh.field.setAsBackgroundMesh(2)

# ---------- Extrude in Z ----------
extrude_out = SetFactory.extrude([(2, domain_surf)], 0, 0, Lz, numElements=[Nz])
print("Extrusion result:", extrude_out)
gmsh.write(os.path.join(script_dir, "debug_output.geo_unrolled"))

SetFactory.synchronize()
exit()
# ---------- Volume ----------
surf_loop = SetFactory.addSurfaceLoop([e[1] for e in extrude_out if e[0] == 2])
SetFactory.addVolume([surf_loop])

# ---------- Mesh Options ----------
gmsh.option.setNumber("Mesh.MeshSizeMin", lc_min)
gmsh.option.setNumber("Mesh.MeshSizeMax", lc_max)
# gmsh.option.setNumber("Mesh.Algorithm", 5)
# gmsh.option.setNumber("Mesh.Algorithm3D", 1)
# gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)

# ---------- Generate and Export ----------
gmsh.model.mesh.generate(3)
gmsh.write(os.path.join(script_dir, "naca4412_field_cgrid3D.msh"))
gmsh.write(os.path.join(script_dir, "naca4412_field_cgrid3D.cgns"))
gmsh.finalize()
