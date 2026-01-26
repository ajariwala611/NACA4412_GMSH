# ------------------- naca4412_cgrid_2D.py -------------------
import gmsh, numpy as np, math, sys, os

# ---------- editable params ----------
c          = 1.0               # chord
R          = 5.0 * c           # upstream semi-circle radius
Lwake      = 10.0 * c          # downstream length
Nu, Nl     = 100, 50           # points on air-foil (top/bottom)
Ntheta     = 20                # resolution of half-circle
lc_wall    = 1.0e-3 * c        # size on air-foil
lc_far     = 2 * c        # far-field size
lc_up      = 2 * c        # upstream size

# -------------------------------------

# --- helper: analytical NACA-4412 with closed TE -------------
def naca4412(x,closed_TE=True):
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
gmsh.model.add("NACA4412_2D")

# --- helper: cosine-distributed spacing ---------------------------
def cosine_spacing(N, start=0.0, end=1.0):
    beta = np.linspace(0, math.pi, N)
    x = 0.5 * (1 - np.cos(beta))   # cosine spacing
    return start + (end - start) * x

# ----- unique trailing-edge point ---------------------------------
pt_TE = gmsh.model.geo.addPoint(c, 0.0, 0.0, lc_wall)  # Trailing edge
pt_LE = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc_wall)  # Leading edge

# Top surface: from TE to LE (excluding TE, including LE)
xU_cos = cosine_spacing(Nu + 1, start=1.0, end=0.0)
xU, yU, _, _ = naca4412(xU_cos)
top_pts = [gmsh.model.geo.addPoint(x*c, y*c, 0, lc_wall) for x, y in zip(xU[1:], yU[1:])]
top_ids = [pt_TE] + top_pts + [pt_LE]

# Bottom surface: from LE to TE (excluding LE, including TE)
xL_cos = cosine_spacing(Nl + 1, start=0.0, end=1.0)
_, _, xL, yL = naca4412(xL_cos)
bot_pts = [gmsh.model.geo.addPoint(x*c, y*c, 0, lc_wall) for x, y in zip(xL[1:], yL[1:])]
bot_ids = [pt_LE] + bot_pts + [pt_TE]

# Build the spline (no duplicate points at TE)
airfoil = gmsh.model.geo.addSpline(top_ids + bot_ids[1:])  # Skip bot_ids[0] (which is LE, already included)

gmsh.model.geo.synchronize()

p1 = gmsh.model.geo.addPoint(0, R, 0, lc_far)          # top-left
p2 = gmsh.model.geo.addPoint(0, -R, 0, lc_far)
arc = gmsh.model.geo.addCircleArc(p1, pt_LE, p2)      # create arc from p1 to p2 through center

p_botOut = gmsh.model.geo.addPoint(Lwake, -R, 0, lc_far)
p_topOut = gmsh.model.geo.addPoint(Lwake,  R, 0, lc_far)

lines = []
lines.append(gmsh.model.geo.addLine(p2, p_botOut))     # bottom edge
lines.append(gmsh.model.geo.addLine(p_botOut, p_topOut))  # vertical outflow
lines.append(gmsh.model.geo.addLine(p_topOut, p1))     # top edge

outer_loop = gmsh.model.geo.addCurveLoop([arc] + lines)       # CCW
 # CW (hole)
inner_loop = gmsh.model.geo.addCurveLoop([-airfoil])  # CW (hole)
surf       = gmsh.model.geo.addPlaneSurface([outer_loop, inner_loop], checkClosed=True)

gmsh.model.geo.synchronize()

Lz = 0.1 * c  # Extrusion length
nLayers = 10  # Number of layers along Z

gmsh.model.geo.extrude(
    [(2, surf)],   # Surface to extrude
    0, 0, Lz,      # Extrusion vector (along Z)
    numElements=[nLayers],
    recombine=True  # Recombine into hexes
)[1][1]
gmsh.model.geo.synchronize()
gmsh.model.geo.removeAllDuplicates()
# Get all surfaces
surfaces = gmsh.model.getEntities(dim=2)
eps = 1e-6
# 1. Airfoil wall (hole boundary) -- usually near origin, small bounding box
surfaces_airfoil = gmsh.model.getEntitiesInBoundingBox(
    -0.1*c, -0.1*c, -eps,
     1.1*c,  0.1*c, Lz+eps,
    2
)

# 2. Inlet (C-grid arc) -- upstream, large radius, at z=0 and z=Lz
surfaces_inlet = gmsh.model.getEntitiesInBoundingBox(
    -R-eps, -R-eps, -eps,
     eps,   R+eps,  Lz+eps,
    2
)

# 3. Outlet (far downstream face)
surfaces_outlet = gmsh.model.getEntitiesInBoundingBox(
    Lwake-eps, -R-eps, -eps,
    Lwake+eps,  R+eps, Lz+eps,
    2
)

# 4. Bottom (y = -R)
surfaces_bottom = gmsh.model.getEntitiesInBoundingBox(
    -R-eps, -R-eps, -eps,
     c+Lwake+eps, -R+eps, Lz+eps,
    2
)

# 5. Top (y = +R)
surfaces_top = gmsh.model.getEntitiesInBoundingBox(
    -R-eps, R-eps, -eps,
     c+Lwake+eps, R+eps, Lz+eps,
    2
)

# 6. Side1 (z = 0)
surfaces_side1 = gmsh.model.getEntitiesInBoundingBox(
    -R-eps, -R-eps, -eps,
     c+Lwake+eps,  R+eps, eps,
    2
)

# 7. Side2 (z = Lz)
surfaces_side2 = gmsh.model.getEntitiesInBoundingBox(
    -R-eps, -R-eps, Lz-eps,
     c+Lwake+eps,  R+eps, Lz+eps,
    2
)

# 8. Fluid volume (after extrusion)
volumes = gmsh.model.getEntities(dim=3)
fluid_tags = [tag for (dim, tag) in volumes]

# ------------- Assign Physical Groups ----------------

gmsh.model.addPhysicalGroup(2, [tag for (dim, tag) in surfaces_airfoil], name="airfoil")
gmsh.model.addPhysicalGroup(2, [tag for (dim, tag) in surfaces_inlet], name="inlet")
gmsh.model.addPhysicalGroup(2, [tag for (dim, tag) in surfaces_outlet], name="outlet")
gmsh.model.addPhysicalGroup(2, [tag for (dim, tag) in surfaces_bottom], name="bottom")
gmsh.model.addPhysicalGroup(2, [tag for (dim, tag) in surfaces_top], name="top")
gmsh.model.addPhysicalGroup(2, [tag for (dim, tag) in surfaces_side1], name="side1")
gmsh.model.addPhysicalGroup(2, [tag for (dim, tag) in surfaces_side2], name="side2")
gmsh.model.addPhysicalGroup(3, fluid_tags, name="fluid")

# Add boundary layer field
# f = gmsh.model.mesh.field
# f.add("BoundaryLayer", 1)
# f.setNumbers(1, "CurvesList", [airfoil])   # Apply to the airfoil spline
# BL_thickness = 0.1 * c         # boundary layer thickness
# progression_ratio = 1.01         # growth ratio
# nbl = 15
# f = gmsh.model.mesh.field
# bl_id = f.add("BoundaryLayer")
# f.setNumbers(bl_id, "CurvesList", [inner_loop])   # Apply to the airfoil spline
# f.setNumber(bl_id, "hwall_n", lc_wall)
# f.setNumber(bl_id, "thickness", BL_thickness)
# f.setNumber(bl_id, "ratio", progression_ratio)
# f.setNumber(bl_id, "NbLayers", nbl)
# f.setNumber(bl_id, "Quads", 1)
# f.setAsBackgroundMesh(bl_id)

# # --- Refinement box field ---
# box_field1 = gmsh.model.mesh.field.add("Box")
# gmsh.model.mesh.field.setNumber(box_field1, "VIn", 0.002)
# gmsh.model.mesh.field.setNumber(box_field1, "VOut", 2)
# gmsh.model.mesh.field.setNumber(box_field1, "Thickness", 0.5)
# gmsh.model.mesh.field.setNumber(box_field1, "XMin", -0.05*c)
# gmsh.model.mesh.field.setNumber(box_field1, "XMax", 1.1*c)
# gmsh.model.mesh.field.setNumber(box_field1, "YMin", -0.025*c)
# gmsh.model.mesh.field.setNumber(box_field1, "YMax", 0.12*c)
# gmsh.model.mesh.field.setAsBackgroundMesh(box_field1)

distance_field_top = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance_field_top, "PointsList", top_ids)

threshold_field_top = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold_field_top, "InField", distance_field_top)
gmsh.model.mesh.field.setNumber(threshold_field_top, "SizeMin", 0.002)
gmsh.model.mesh.field.setNumber(threshold_field_top, "SizeMax", 2)
gmsh.model.mesh.field.setNumber(threshold_field_top, "DistMin", 0.015)
gmsh.model.mesh.field.setNumber(threshold_field_top, "DistMax", 0.025*c)

distance_field_bot = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance_field_bot, "PointsList", bot_ids)

threshold_field_bot = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold_field_bot, "InField", distance_field_bot)
gmsh.model.mesh.field.setNumber(threshold_field_bot, "SizeMin", 0.005)
gmsh.model.mesh.field.setNumber(threshold_field_bot, "SizeMax", 2)
gmsh.model.mesh.field.setNumber(threshold_field_bot, "DistMin", 0.01)
gmsh.model.mesh.field.setNumber(threshold_field_bot, "DistMax", 0.05*c)

min_field = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field_top, threshold_field_bot])
gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

gmsh.model.geo.synchronize()
# --- Generate tetrahedral mesh (default) ---
gmsh.option.setNumber("Mesh.ElementOrder", 2)      
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2) 
gmsh.option.setNumber("Mesh.Algorithm", 1) 
# gmsh.option.setNumber("Mesh.Algorithm3D", 4)
gmsh.option.setNumber("Mesh.RecombineAll", 1)  # recombine tri → quad
gmsh.option.setNumber("Mesh.RecombinationAlgorithm",   3)  #
# gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)  # simple subdivision


gmsh.model.mesh.generate(3)

# 4. write & show ---------------------------------------------
# Get the absolute directory where *this* Python script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build full paths
mesh_filename = os.path.join(script_dir, "naca4412_cgrid2D_quad_3D.msh")
cgns_filename = os.path.join(script_dir, "naca4412_cgrid2D_quad_3D.cgns")


# Then write
gmsh.write(mesh_filename)
gmsh.write(cgns_filename)
if "-nopopup" not in sys.argv:
    gmsh.fltk.run()
gmsh.finalize()
# --------------------------------------------------------------
