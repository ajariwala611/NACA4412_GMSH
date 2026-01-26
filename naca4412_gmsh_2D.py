# ------------------- naca4412_cgrid_2D.py -------------------
import gmsh, numpy as np, math, sys, os

# ---------- editable params ----------
c          = 1.0               # chord
R          = 5.0 * c           # upstream semi-circle radius
Lwake      = 10.0 * c          # downstream length
Nu, Nl     = 100, 50           # points on air-foil (top/bottom)
Ntheta     = 20                # resolution of half-circle
lc_wall    = 1.0e-3 * c        # size on air-foil
lc_far     = 2.1 * c        # far-field size
lc_up      = 2.1 * c        # upstream size

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

# Get the airfoil curve tag (from your earlier definition)
airfoil_curve = airfoil  # This is your airfoil spline (curve tag)



# 2. C-grid outer boundary (CCW) -------------------------------
theta = np.linspace(math.pi/2, 3*math.pi/2, Ntheta)  # 90°→270°, CCW
semi   = [gmsh.model.geo.addPoint(R*math.cos(t),
                                  R*math.sin(t), 0, lc_up)
          for t in theta]
p_botOut = gmsh.model.geo.addPoint(c+Lwake, -R, 0, lc_far)
p_topOut = gmsh.model.geo.addPoint(c+Lwake,  R, 0, lc_far)

lines = []
# a) semi-circle arcs  (top→bottom)
for i in range(len(semi)-1):
    lines.append(gmsh.model.geo.addLine(semi[i], semi[i+1]))
# b) bottom edge  (0,-R) → (c+L,-R)
lines.append(gmsh.model.geo.addLine(semi[-1], p_botOut))
# c) vertical outflow  (c+L,-R) → (c+L,+R)
lines.append(gmsh.model.geo.addLine(p_botOut, p_topOut))
# d) top edge  (c+L,+R) → (0,+R)
lines.append(gmsh.model.geo.addLine(p_topOut, semi[0]))

outer_loop = gmsh.model.geo.addCurveLoop(lines)       # CCW
inner_loop = gmsh.model.geo.addCurveLoop([-airfoil])  # CW (hole)
surf       = gmsh.model.geo.addPlaneSurface([outer_loop, inner_loop])



# 3. mesh options: quad-only, fast -----------------------------
gmsh.model.geo.synchronize()
gmsh.model.mesh.removeDuplicateNodes()


# Add boundary layer field
# f = gmsh.model.mesh.field
# f.add("BoundaryLayer", 1)
# f.setNumbers(1, "CurvesList", [airfoil])   # Apply to the airfoil spline
# BL_thickness = 0.1 * c         # boundary layer thickness
# progression_ratio = 1.01         # growth ratio
# nbl = 15
# # Wall-normal control parameters
# f.setNumber(1, "hwall_n", lc_wall)         # fine spacing near the wall
# f.setNumber(1, "thickness", BL_thickness)          # total boundary layer thickness (say 10% chord)
# f.setNumber(1, "ratio", progression_ratio)                # how fast to grow (e.g., 10% per layer)
# f.setNumber(1, "NbLayers", nbl)              # number of wall-normal layers
# f.setNumber(1, "Quads", 1)             # 

# # Activate this field as the background mesh
# f.setAsBackgroundMesh(1)

# # --- Refinement box field ---
# box_field1 = gmsh.model.mesh.field.add("Box")
# gmsh.model.mesh.field.setNumber(box_field1, "VIn", 0.001)
# gmsh.model.mesh.field.setNumber(box_field1, "VOut", 2)
# gmsh.model.mesh.field.setNumber(box_field1, "Thickness", 1)
# gmsh.model.mesh.field.setNumber(box_field1, "XMin", -0.1*c)
# gmsh.model.mesh.field.setNumber(box_field1, "XMax", 1.25*c)
# gmsh.model.mesh.field.setNumber(box_field1, "YMin", -0.025*c)
# gmsh.model.mesh.field.setNumber(box_field1, "YMax", 0.15*c)


# gmsh.model.mesh.field.setAsBackgroundMesh(box_field1)

# --- Generate tetrahedral mesh (default) ---
gmsh.option.setNumber("Mesh.ElementOrder", 2)      
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2) 
gmsh.option.setNumber("Mesh.Algorithm",      1)  # quad-delaunay (fast)
gmsh.option.setNumber("Mesh.RecombineAll",   1)  # recombine tri → quad
gmsh.option.setNumber("Mesh.RecombinationAlgorithm",   1)  #
# gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)  # simple subdivision
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
