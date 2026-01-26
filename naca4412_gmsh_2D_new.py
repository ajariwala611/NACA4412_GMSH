# ------------------- naca4412_cgrid_2D.py -------------------
import gmsh, numpy as np, math, sys, os

# ---------- editable params ----------
c          = 1.0               # chord
R          = 5.0 * c           # upstream semi-circle radius
Lwake      = 10.0 * c          # downstream length
Nu, Nl     = 500, 100           # points on air-foil (top/bottom)
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
pt_TE = gmsh.model.geo.addPoint(c, 0.0, 0.0, lc_wall)  # one TE point

# generate top surface (cosine spacing from TE (x=1) back to LE (x=0))
xU_cos = cosine_spacing(Nu + 1, start=1.0, end=0.0)
xU, yU, _, _ = naca4412(xU_cos)

top_ids = [pt_TE] + [gmsh.model.geo.addPoint(x*c, y*c, 0, lc_wall)
                     for x, y in zip(xU[1:], yU[1:])]

# generate bottom surface (cosine spacing from LE (x=0) to TE (x=1))
xL_cos = cosine_spacing(Nl + 1, start=0.0, end=1.0)
_, _, xL, yL = naca4412(xL_cos)

bot_ids = [gmsh.model.geo.addPoint(x*c, y*c, 0, lc_wall)
           for x, y in zip(xL[1:], yL[1:])] + [pt_TE]

airfoil = gmsh.model.geo.addSpline(top_ids + bot_ids)

gmsh.model.geo.synchronize()

p1 = gmsh.model.geo.addPoint(0, R, 0, lc_far)          # top-left
p2 = gmsh.model.geo.addPoint(0, -R, 0, lc_far)
center = gmsh.model.geo.addPoint(0, 0, 0, lc_up)       # center of upstream arc
arc = gmsh.model.geo.addCircleArc(p1, center, p2)      # create arc from p1 to p2 through center

p_botOut = gmsh.model.geo.addPoint(c+Lwake, -R, 0, lc_far)
p_topOut = gmsh.model.geo.addPoint(c+Lwake,  R, 0, lc_far)

lines = []
lines.append(gmsh.model.geo.addLine(p2, p_botOut))     # bottom edge
lines.append(gmsh.model.geo.addLine(p_botOut, p_topOut))  # vertical outflow
lines.append(gmsh.model.geo.addLine(p_topOut, p1))     # top edge

outer_loop = gmsh.model.geo.addCurveLoop([arc] + lines)       # CCW
inner_loop = gmsh.model.geo.addCurveLoop([airfoil])  # CW (hole)
surf       = gmsh.model.geo.addPlaneSurface([outer_loop, inner_loop])

gmsh.model.geo.synchronize()

# # Add boundary layer field
# f = gmsh.model.mesh.field
# f.add("BoundaryLayer", 1)
# f.setNumbers(1, "CurvesList", [airfoil])   # Apply to the airfoil spline
# BL_thickness = 0.01 * c         # boundary layer thickness
# progression_ratio = 1.1         # growth ratio
# nbl = 15
# f = gmsh.model.mesh.field
# bl_id = f.add("BoundaryLayer")
# f.setNumbers(bl_id, "CurvesList", [airfoil])   # Apply to the airfoil spline
# f.setNumber(bl_id, "hwall_n", 0.002)
# f.setNumber(bl_id, "thickness", BL_thickness)
# f.setNumber(bl_id, "ratio", progression_ratio)
# f.setNumber(bl_id, "NbLayers", nbl)
# f.setNumber(bl_id, "Quads", 1)
# f.setAsBoundaryLayer(bl_id)

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
gmsh.model.mesh.field.setNumber(threshold_field_top, "SizeMin", 0.001)
gmsh.model.mesh.field.setNumber(threshold_field_top, "SizeMax", 2)
gmsh.model.mesh.field.setNumber(threshold_field_top, "DistMin", 0.02)
gmsh.model.mesh.field.setNumber(threshold_field_top, "DistMax", 0.02*c)

distance_field_bot = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance_field_bot, "PointsList", bot_ids)

threshold_field_bot = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold_field_bot, "InField", distance_field_bot)
gmsh.model.mesh.field.setNumber(threshold_field_bot, "SizeMin", 0.005)
gmsh.model.mesh.field.setNumber(threshold_field_bot, "SizeMax", 2)
gmsh.model.mesh.field.setNumber(threshold_field_bot, "DistMin", 0.01)
gmsh.model.mesh.field.setNumber(threshold_field_bot, "DistMax", 0.05*c)

box_field1 = gmsh.model.mesh.field.add("Box")
gmsh.model.mesh.field.setNumber(box_field1, "VIn", 0.002)
gmsh.model.mesh.field.setNumber(box_field1, "VOut", 2)
gmsh.model.mesh.field.setNumber(box_field1, "Thickness", 0.1)
gmsh.model.mesh.field.setNumber(box_field1, "XMin", 0.9*c)
gmsh.model.mesh.field.setNumber(box_field1, "XMax", 1.2*c)
gmsh.model.mesh.field.setNumber(box_field1, "YMin", -0.01*c)
gmsh.model.mesh.field.setNumber(box_field1, "YMax", 0.05*c)
gmsh.model.mesh.field.setAsBackgroundMesh(box_field1)

min_field = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field_top, threshold_field_bot, box_field1])
# gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [box_field1])
gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

gmsh.model.geo.synchronize()
# --- Generate tetrahedral mesh (default) ---
gmsh.option.setNumber("Mesh.ElementOrder", 2)      
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2) 
# gmsh.model.mesh.optimize('HighOrderFastCurving')
# gmsh.model.mesh.optimize('HighOrder')


gmsh.option.setNumber("Mesh.Algorithm", 1) 
gmsh.option.setNumber("Mesh.RecombineAll", 1)  # recombine tri → quad
gmsh.option.setNumber("Mesh.RecombinationAlgorithm",   3) 
# gmsh.option.setNumber("Mesh.RecombineOptimizeTopology",   200) 


gmsh.model.mesh.generate(2)

# 4. write & show ---------------------------------------------
# Get the absolute directory where *this* Python script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build full paths
mesh_filename = os.path.join(script_dir, "naca4412_cgrid2D_quad.msh")
cgns_filename = os.path.join(script_dir, "naca4412_cgrid2D_quad.cgns")


# Then write
gmsh.write(mesh_filename)
gmsh.write(cgns_filename)
if "-nopopup" not in sys.argv:
    gmsh.fltk.run()
gmsh.finalize()
# --------------------------------------------------------------
