# ------------------- naca4412_cgrid_2D.py -------------------
import gmsh, numpy as np, math, sys, os

# ---------- editable params ----------
c          = 1.0               # chord
R          = 5.0 * c           # upstream semi-circle radius
Lwake      = 10.0 * c          # downstream length
Nu, Nl     = 1000, 500           # points on air-foil (top/bottom)
Ntheta     = 20                # resolution of half-circle
lc_wall    = 1.0e-3 * c        # size on air-foil
lc_far     = 2 * c        # far-field size
lc_up      = 2 * c        # upstream size
Lz         = 0.1 * c  # Extrusion length
nLayers    = 40  # Number of layers along Z
# -------------------------------------

# --- helper: analytical NACA-4412 with closed TE -------------
def naca4412(x, closed_TE=True):
    m, p, t = 0.04, 0.4, 0.12
    yt = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - (0.1036 if closed_TE else 0.1015)*x**4)
    yc  = np.where(x < p, m/p**2*(2*p*x - x**2),
                   m/(1-p)**2*((1-2*p)+2*p*x - x**2))
    dyc = np.where(x < p, 2*m/p**2*(p-x),
                   2*m/(1-p)**2*(p-x))
    th  = np.arctan(dyc)
    xu, yu = x - yt*np.sin(th), yc + yt*np.cos(th)
    xl, yl = x + yt*np.sin(th), yc - yt*np.cos(th)
    return xu, yu, xl, yl

# --- gmsh initialisation -------------------------------------
gmsh.initialize()
gmsh.model.add("NACA4412_3D_extrude")

# --- helper: cosine-distributed spacing ---------------------------
def cosine_spacing(N, start=0.0, end=1.0):
    beta = np.linspace(0, math.pi, N)
    x = 0.5 * (1 - np.cos(beta))   # cosine spacing
    return start + (end - start) * x

# --- Trailing and Leading Edge Points ---
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

# --- C-grid outer boundary ---
p1 = gmsh.model.occ.addPoint(0, R, 0, lc_far)          # top-left
p2 = gmsh.model.occ.addPoint(0, -R, 0, lc_far)
arc = gmsh.model.occ.addCircleArc(p2, pt_LE, p1)

    # create arc from p1 to p2 through center

p_botOut = gmsh.model.occ.addPoint(Lwake, -R, 0, lc_far)
p_topOut = gmsh.model.occ.addPoint(Lwake,  R, 0, lc_far)

line_bot = gmsh.model.occ.addLine(p2, p_botOut)     # bottom edge
line_out = gmsh.model.occ.addLine(p_botOut, p_topOut)  # vertical outflow
line_top = gmsh.model.occ.addLine(p_topOut, p1)     # top edge

# Outer wire (CCW)
outer_wire = gmsh.model.occ.addWire([arc, line_bot, line_out, line_top])

# Plane surface with hole (airfoil)
surf = gmsh.model.occ.addPlaneSurface([outer_wire, airfoil_loop])

gmsh.model.occ.synchronize()

gmsh.model.occ.extrude(
    [(2, surf)],   # Surface to extrude
    0, 0, Lz,      # Extrusion vector (along Z)
    numElements=[nLayers],
    recombine=True  # Recombine into hexes
)[1][1]
gmsh.model.occ.synchronize()
gmsh.model.occ.removeAllDuplicates()
# Get all surfaces
surfaces = gmsh.model.getEntities(dim=2)
volumes = gmsh.model.getEntities(dim=3)
fluid_tags = [tag for (dim, tag) in volumes]
tol = 1e-6
eps = 1e-6

inlet = []
outlet = []
airfoil = []
side1 = []
side2 = []
top = []
bottom = []

for (dim, tag) in surfaces:
    com = gmsh.model.occ.getCenterOfMass(dim, tag)

    # Outlet: x = max (box far downstream face)
    if abs(com[0] - Lwake) < tol:
        outlet.append(tag)
    # Side 1: y = min
    elif abs(com[1] + R) < tol:
        bottom.append(tag)
    # Side 2: y = max
    elif abs(com[1] - R) < tol:
        top.append(tag)
    # Top: z = Lz
    elif abs(com[2] - Lz) < tol:
        side2.append(tag)
    # Bottom: z = 0
    elif abs(com[2] - 0) < tol:
        side1.append(tag)

surfaces_airfoil = gmsh.model.getEntitiesInBoundingBox(
    -eps, - Lz - eps, - eps,
    c + eps, Lz + eps,  Lz + eps,
    2
)

surfaces_inlet = gmsh.model.getEntitiesInBoundingBox(
    -R - eps, -R - eps, - eps,
    eps, R + eps, Lz + eps,
    2
)
gmsh.model.addPhysicalGroup(2, [tag for (dim, tag) in surfaces_inlet], name="inlet")
if outlet:
    gmsh.model.addPhysicalGroup(2, outlet, name="outlet")
gmsh.model.addPhysicalGroup(2, [tag for (dim, tag) in surfaces_airfoil], name="airfoil")
if top:
    gmsh.model.addPhysicalGroup(2, top, name="top")
if bottom:
    gmsh.model.addPhysicalGroup(2, bottom, name="bottom")
if side1:
    gmsh.model.addPhysicalGroup(2, side1, name="side1")
if side2:
    gmsh.model.addPhysicalGroup(2, side2, name="side2")
if fluid_tags:
    gmsh.model.addPhysicalGroup(3, fluid_tags, name="fluid")

# # --- Refinement box field ---
box_field1 = gmsh.model.mesh.field.add("Box")
gmsh.model.mesh.field.setNumber(box_field1, "VIn", 0.004)
gmsh.model.mesh.field.setNumber(box_field1, "VOut", 2)
gmsh.model.mesh.field.setNumber(box_field1, "Thickness", 0.1)
gmsh.model.mesh.field.setNumber(box_field1, "XMin", 0.9*c)
gmsh.model.mesh.field.setNumber(box_field1, "XMax", 1.15*c)
gmsh.model.mesh.field.setNumber(box_field1, "YMin", -0.01*c)
gmsh.model.mesh.field.setNumber(box_field1, "YMax", 0.05*c)
gmsh.model.mesh.field.setNumber(box_field1, "ZMin", 0)
gmsh.model.mesh.field.setNumber(box_field1, "ZMax", Lz)
gmsh.model.mesh.field.setAsBackgroundMesh(box_field1)

distance_field_top = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance_field_top, "SurfacesList", [surfaces_airfoil[0][1]])

threshold_field_top = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold_field_top, "InField", distance_field_top)
gmsh.model.mesh.field.setNumber(threshold_field_top, "SizeMin", 0.002)
gmsh.model.mesh.field.setNumber(threshold_field_top, "SizeMax", 2)
gmsh.model.mesh.field.setNumber(threshold_field_top, "DistMin", 0.015)
gmsh.model.mesh.field.setNumber(threshold_field_top, "DistMax", 0.015*c)

distance_field_bot = gmsh.model.mesh.field.add("Distance")
# gmsh.model.mesh.field.setNumbers(distance_field_bot, "PointsList", lower_pts)
gmsh.model.mesh.field.setNumbers(distance_field_bot, "SurfacesList", [surfaces_airfoil[1][1]])

threshold_field_bot = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold_field_bot, "InField", distance_field_bot)
gmsh.model.mesh.field.setNumber(threshold_field_bot, "SizeMin", 0.005)
gmsh.model.mesh.field.setNumber(threshold_field_bot, "SizeMax", 2)
gmsh.model.mesh.field.setNumber(threshold_field_bot, "DistMin", 0.01)
gmsh.model.mesh.field.setNumber(threshold_field_bot, "DistMax", 0.05*c)

min_field = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field_top, threshold_field_bot, box_field1])
gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

gmsh.model.occ.synchronize()
gmsh.option.setNumber("Mesh.ElementOrder", 2)      
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2) 

gmsh.option.setNumber("Mesh.Algorithm", 1) 
gmsh.option.setNumber("Mesh.RecombineAll", 1)  # recombine tri → quad
gmsh.option.setNumber("Mesh.Recombine3DAll", 1)
gmsh.option.setNumber("Mesh.Recombine3DLevel", 0)
gmsh.option.setNumber("Mesh.RecombinationAlgorithm",   3)  #

gmsh.model.mesh.generate(3)

# 4. write & show ---------------------------------------------
# Get the absolute directory where *this* Python script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build full paths
mesh_filename = os.path.join(script_dir, "naca4412_cgrid_quad_3D_extrude.msh")
cgns_filename = os.path.join(script_dir, "naca4412_cgrid_quad_3D_extrude.cgns")


# Then write
gmsh.write(mesh_filename)
gmsh.write(cgns_filename)
if "-nopopup" not in sys.argv:
    gmsh.fltk.run()
gmsh.finalize()
# --------------------------------------------------------------
