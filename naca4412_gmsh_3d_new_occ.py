import gmsh, numpy as np, math, os

# ---------- Parameters ----------
c = 1.0
R = 5.0 * c
Lwake = 10.0 * c
Nu, Nl = 100, 50
Ntheta = 20
lc_wall = 5e-3 * c
lc_far = 4.0 * c
lc_up = 4.0 * c
Lz = 0.1 * c
Lz_res = 10
Nz_outer = 3
Nz_inner = 10
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
gmsh.model.add("NACA4412_OCC")

# ---------- Airfoil Geometry (OCC) ----------
# Cosine spacing for both upper and lower surfaces
xU_cos = cosine_spacing(Nu + 1, 1.0, 0.0)
xL_cos = cosine_spacing(Nl + 1, 0.0, 1.0)
xu, yu, _, _ = naca4412(xU_cos)
_, _, xl, yl = naca4412(xL_cos)

# Create TE and LE points first to be reused
pt_TE = gmsh.model.occ.addPoint(c, 0.0, 0.0, lc_wall)  # x=1
pt_LE = gmsh.model.occ.addPoint(0.0, 0.0, 0.0, lc_wall)  # x=0

# Build upper surface points from TE to LE (exclude TE, LE)
upper_pts = [pt_TE] + [gmsh.model.occ.addPoint(x*c, y*c, 0, lc_wall)
                       for x, y in zip(xu[1:-1], yu[1:-1])] + [pt_LE]

# Build lower surface points from LE to TE (exclude LE, TE)
lower_pts = [pt_LE] + [gmsh.model.occ.addPoint(x*c, y*c, 0, lc_wall)
                       for x, y in zip(xl[1:-1], yl[1:-1])] + [pt_TE]

# Create splines
curve_upper = gmsh.model.occ.addSpline(upper_pts)
curve_lower = gmsh.model.occ.addSpline(lower_pts)

# Create closed wire
airfoil_loop = gmsh.model.occ.addWire([curve_upper, curve_lower], checkClosed=True)

# ---------- Outer Boundary (C-grid with semicircle) ----------
theta = np.linspace(math.pi/2, 3*math.pi/2, Ntheta)
semi = [gmsh.model.occ.addPoint(R*math.cos(t), R*math.sin(t), 0, lc_up) for t in theta]
p_botOut = gmsh.model.occ.addPoint(c+Lwake, -R, 0, lc_far)
p_topOut = gmsh.model.occ.addPoint(c+Lwake,  R, 0, lc_far)

lines = [gmsh.model.occ.addLine(semi[i], semi[i+1]) for i in range(len(semi)-1)]
lines += [
    gmsh.model.occ.addLine(semi[-1], p_botOut),
    gmsh.model.occ.addLine(p_botOut, p_topOut),
    gmsh.model.occ.addLine(p_topOut, semi[0])
]
outer_loop = gmsh.model.occ.addWire(lines, checkClosed=True)

# ---------- Surfaces ----------
left_surf = gmsh.model.occ.addPlaneSurface([outer_loop, airfoil_loop])


# ---------- Extrusions ----------
outer_ext = gmsh.model.occ.extrude([(2, left_surf)], 0, 0, Lz, numElements=[Nz_outer])
print("Outer extrusion:", outer_ext)
gmsh.model.occ.synchronize()  # or gmsh.model.geo.synchronize()

# Get all the surfaces tags 
right_surf = outer_ext[0][1]
outer_shell = [tag for (dim, tag) in outer_ext[2:-2] if dim == 2]
airfoil_wall_tags = [tag for (dim, tag) in outer_ext[-2:] if dim == 2]
gmsh.model.occ.mesh.setSize([(2,airfoil_wall_tags[0]),(2,airfoil_wall_tags[1])], Lz/Lz_res)
print("Airfoil wall tags:", airfoil_wall_tags)
print("Outer shell tags:", outer_shell)
print("Right surface tag:", right_surf)
gmsh.write(os.path.join(script_dir, "debug_output.geo_unrolled"))
# gmsh.finalize()
# exit()

# ---------- Volume Construction ----------
# top_outer = [e[1] for e in outer_ext if e[0] == 2][-1]
# top_inner = [e[1] for e in inner_ext if e[0] == 2][-1]
# side_outer = [e[1] for e in outer_ext if e[0] == 2 and e[1] != top_outer]
# side_inner = [e[1] for e in inner_ext if e[0] == 2 and e[1] != top_inner]
all_surfaces = [left_surf, right_surf] + outer_shell + airfoil_wall_tags
surf_loop = gmsh.model.occ.addSurfaceLoop(all_surfaces)
volume = gmsh.model.occ.addVolume([surf_loop])


# ---------- Synchronize and Mesh ----------
gmsh.model.occ.synchronize()
# gmsh.option.setNumber("Mesh.RecombineAll", 0)
gmsh.option.setNumber("Mesh.MeshSizeMin", lc_wall)
gmsh.option.setNumber("Mesh.MeshSizeMax", lc_far)
gmsh.option.setNumber("Mesh.Algorithm",      5)  # quad-delaunay (fast)
gmsh.option.setNumber("Mesh.Algorithm3D",      1)  
# gmsh.option.setNumber("Mesh.RecombineAll",   1)  # recombine tri → quad
# gmsh.option.setNumber("Mesh.RecombinationAlgorithm",   1)  #


gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)  # simple subdivision
gmsh.model.mesh.generate(3)

# ---------- Write Output ----------

gmsh.write(os.path.join(script_dir, "naca4412_occ_cgrid3D_tet.msh"))
gmsh.write(os.path.join(script_dir, "naca4412_occ_cgrid3D_tet.cgns"))
gmsh.finalize()
