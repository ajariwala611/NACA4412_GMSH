import gmsh, numpy as np, math, os
import sys

# ---------- Parameters ----------
c = 1.0
R = 5 * c
Lwake = 10.0 * c
Nu, Nl = 500, 100
lc_wall = 5e-3 * c
Lz = 0.1 * c
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
# # Cosine spacing for both upper and lower surfaces
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
airfoil_surface = gmsh.model.occ.addPlaneSurface([airfoil_loop])
airfoil_volume = gmsh.model.occ.extrude([(2, airfoil_surface)], 0, 0, Lz)[1][1]  # [1][1] gives the volume tag

# --- 1. Rectangular box ---
box_x_min, box_x_max = 0, Lwake
box_y_min, box_y_max = -R, R
box_z_min, box_z_max = 0, Lz

box = gmsh.model.occ.addBox(
    box_x_min, box_y_min, box_z_min,
    box_x_max - box_x_min,
    box_y_max - box_y_min,
    box_z_max - box_z_min
)
# --- 2. Semi-cylinder at the inlet (rear half) ---
semi_cylinder = gmsh.model.occ.addCylinder(
    0, 0, 0,   # Center of first circular face (y = +R)
    0, 0, Lz,        # Axis vector: along z direction, thickness Lz
    R,          # R
    -1,              # Tag (auto)
    math.pi          # Angular opening (pi for rear half)
)
theta = math.pi / 2  # Example rotation
gmsh.model.occ.rotate([(3, semi_cylinder)],
    0, 0, 0,    # Center of rotation
    0, 0, 1,     # Axis of rotation (z-axis)
    theta        # Rotation angle
)
gmsh.model.occ.synchronize()

# --- Fuse the two volumes ---
fused_volumes, fused_map = gmsh.model.occ.fuse(
    [(3, box)], [(3, semi_cylinder)]
)
gmsh.model.occ.synchronize()

fluid_domain, _ = gmsh.model.occ.cut(fused_volumes, [(3, airfoil_volume)])
gmsh.model.occ.synchronize()
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


# gmsh.write(os.path.join(script_dir, "debug_output.geo_unrolled"))

# print(top[0])
# exit()
gmsh.model.mesh.setTransfiniteSurface(surfaces_airfoil[0][1])
gmsh.model.mesh.setTransfiniteSurface(surfaces_airfoil[1][1])
gmsh.model.mesh.setTransfiniteSurface(surfaces_inlet[0][1])
gmsh.model.mesh.setTransfiniteSurface(top[0])
gmsh.model.mesh.setTransfiniteSurface(bottom[0])
gmsh.model.mesh.setTransfiniteSurface(outlet[0])

# Set periodicity in z-direction
translation = [1, 0, 0, 0,
               0, 1, 0, 0,
               0, 0, 1, Lz,
               0, 0, 0, 1]

# gmsh.model.mesh.setPeriodic(2, side1_sorted, side2_sorted, translation)


# --- Generate tetrahedral mesh (default) ---
gmsh.option.setNumber("Mesh.ElementOrder", 2)      # Quadratic elements
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2) 


# --- Set mesh options ---
gmsh.option.setNumber("Mesh.Algorithm", 1)
gmsh.option.setNumber("Mesh.Algorithm3D", 1)
gmsh.option.setNumber("Mesh.RecombineAll", 1)
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)  
gmsh.option.setNumber("Mesh.Recombine3DAll", 1)
gmsh.option.setNumber("Mesh.Recombine3DLevel", 0)
gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)
# gmsh.option.setNumber("Mesh.RefineSteps", 100)
# gmsh.option.setNumber("Mesh.RecombineOptimizeTopology", 10)

# # --- Refinement box field ---
box_field1 = gmsh.model.mesh.field.add("Box")
gmsh.model.mesh.field.setNumber(box_field1, "VIn", 0.005)
gmsh.model.mesh.field.setNumber(box_field1, "VOut", 10)
gmsh.model.mesh.field.setNumber(box_field1, "Thickness", 1)
gmsh.model.mesh.field.setNumber(box_field1, "XMin", 0.9*c)
gmsh.model.mesh.field.setNumber(box_field1, "XMax", 1.2*c)
gmsh.model.mesh.field.setNumber(box_field1, "YMin", -0.01*c)
gmsh.model.mesh.field.setNumber(box_field1, "YMax", 0.05*c)
gmsh.model.mesh.field.setNumber(box_field1, "ZMin", 0)
gmsh.model.mesh.field.setNumber(box_field1, "ZMax", Lz)

# gmsh.model.mesh.field.setAsBackgroundMesh(box_field1)

# dist_field = gmsh.model.mesh.field.add("Distance")
# gmsh.model.mesh.field.setNumbers(dist_field, "FacesList", [airfoil_surface])

# thresh_field = gmsh.model.mesh.field.add("Threshold")
# gmsh.model.mesh.field.setNumber(thresh_field, "InField", dist_field)
# gmsh.model.mesh.field.setNumber(thresh_field, "SizeMin", 0.005)  # Fine near airfoil
# gmsh.model.mesh.field.setNumber(thresh_field, "SizeMax", 2)     # Coarse far away
# gmsh.model.mesh.field.setNumber(thresh_field, "DistMin", 0.0)
# gmsh.model.mesh.field.setNumber(thresh_field, "DistMax", 0.5)    # Transition distance

# Combine with Min field if needed
# min_field = gmsh.model.mesh.field.add("Min")
# gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [box_field1])
# gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

distance_field_top = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance_field_top, "SurfacesList", [surfaces_airfoil[0][1]])

threshold_field_top = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold_field_top, "InField", distance_field_top)
gmsh.model.mesh.field.setNumber(threshold_field_top, "SizeMin", 0.005)
gmsh.model.mesh.field.setNumber(threshold_field_top, "SizeMax", 4)
gmsh.model.mesh.field.setNumber(threshold_field_top, "DistMin", 0.015)
gmsh.model.mesh.field.setNumber(threshold_field_top, "DistMax", 0.015*c)

distance_field_bot = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance_field_bot, "SurfacesList", [surfaces_airfoil[1][1]])

threshold_field_bot = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold_field_bot, "InField", distance_field_bot)
gmsh.model.mesh.field.setNumber(threshold_field_bot, "SizeMin", 0.01)
gmsh.model.mesh.field.setNumber(threshold_field_bot, "SizeMax", 4)
gmsh.model.mesh.field.setNumber(threshold_field_bot, "DistMin", 0.01)
gmsh.model.mesh.field.setNumber(threshold_field_bot, "DistMax", 0.05*c)

min_field = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field_top, threshold_field_bot, box_field1])
gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

gmsh.model.mesh.generate(3)

# ---------- Write Output ----------
gmsh.write(os.path.join(script_dir, "naca4412_occ_cgrid3D_10c.msh"))
gmsh.write(os.path.join(script_dir, "naca4412_occ_cgrid3D_10c.cgns"))


if '-nopopup' not in sys.argv:
    gmsh.fltk.run()
gmsh.finalize()
