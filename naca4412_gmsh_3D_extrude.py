# ------------------- naca4412_cgrid_2D.py -------------------
import gmsh, numpy as np, math, sys, os

# ---------- editable params ----------
c          = 1.0               # chord
R          = 5.0 * c           # upstream semi-circle radius
Lwake      = 10.0 * c          # downstream length
Nu, Nl     = 100, 50           # points on air-foil (top/bottom)
Ntheta     = 20                # resolution of half-circle
lc_wall    = 7.0e-3 * c        # size on air-foil
lc_far     = 2 * c        # far-field size
lc_up      = 5 * c        # upstream size
BL_thickness = 0.5 * c         # boundary layer thickness
progression_ratio = 1.01         # growth ratio
Lz_span = 0.1 * c     # spanwise extent
Nz_span = 6          # number of elements along span (you can adjust)
script_dir = os.path.dirname(os.path.abspath(__file__))

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
# Airfoil wire
xU_cos = cosine_spacing(Nu + 1, 1.0, 0.0)
xL_cos = cosine_spacing(Nl + 1, 0.0, 1.0)
xu, yu, _, _ = naca4412(xU_cos)
_, _, xl, yl = naca4412(xL_cos)

pt_TE = gmsh.model.geo.addPoint(c, 0.0, 0.0, lc_wall)
pt_LE = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc_wall)

upper_pts = [pt_TE] + [gmsh.model.geo.addPoint(x * c, y * c, 0, lc_wall) for x, y in zip(xu[1:-1], yu[1:-1])] + [pt_LE]
lower_pts = [pt_LE] + [gmsh.model.geo.addPoint(x * c, y * c, 0, lc_wall) for x, y in zip(xl[1:-1], yl[1:-1])] + [pt_TE]

curve_upper = gmsh.model.geo.addSpline(upper_pts)
curve_lower = gmsh.model.geo.addSpline(lower_pts)
# airfoil = gmsh.model.geo.addWire([curve_upper, curve_lower], checkClosed=True)
airfoil = gmsh.model.geo.addCurveLoop([curve_upper, curve_lower])

# Add boundary layer field
# f = gmsh.model.mesh.field
# f.add("BoundaryLayer", 1)
# f.setNumbers(1, "CurvesList", [airfoil])   # Apply to the airfoil spline

# # Wall-normal control parameters
# f.setNumber(1, "hwall_n", lc_wall)         # fine spacing near the wall
# f.setNumber(1, "thickness", BL_thickness)          # total boundary layer thickness (say 10% chord)
# f.setNumber(1, "ratio", progression_ratio)                # how fast to grow (e.g., 10% per layer)
# f.setNumber(1, "NbLayers", 20)              # number of wall-normal layers

# # Activate this field as the background mesh
# f.setAsBackgroundMesh(1)

gmsh.model.geo.synchronize()


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

# outer_loop = gmsh.model.geo.addWire(lines, checkClosed=True)
outer_loop = gmsh.model.geo.addCurveLoop(lines)       # CCW
surf       = gmsh.model.geo.addPlaneSurface([outer_loop, airfoil])

# 3. mesh options: quad-only, fast -----------------------------
gmsh.model.geo.synchronize()
gmsh.model.mesh.removeDuplicateNodes()

# Extrude the 2D surface "surf" into 3D
# extruded = gmsh.model.geo.extrude([(2, surf)], 
#                                   0, 0, Lz_span,numElements=[Nz_span],
#                                   recombine=True)  # recombine to hex
extruded = gmsh.model.geo.extrude([(2, surf)], 0, 0, Lz_span)  
airfoil_surface = extruded[-2:]


N = 20 # number of layers
r = 1.05 # ratio
d = [5e-4] # thickness of first layer
for i in range(1, N): d.append(d[-1] + d[0] * r**i)
d = [-dd for dd in d]
exdbl = gmsh.model.geo.extrudeBoundaryLayer(airfoil_surface,[1] * N, d, True)
print(exdbl)
gmsh.model.geo.synchronize()  # Important: synchronize before extrusion

# gmsh.model.mesh.field.add("Box", 1)
# gmsh.model.mesh.field.setNumber(1, "VIn", 0.01)
# gmsh.model.mesh.field.setNumber(1, "VOut", 2)
# gmsh.model.mesh.field.setNumber(1, "XMin", -0.1)
# gmsh.model.mesh.field.setNumber(1, "XMax", 1.2)
# gmsh.model.mesh.field.setNumber(1, "YMin", -0.05)
# gmsh.model.mesh.field.setNumber(1, "YMax", 0.2)
# gmsh.model.mesh.field.setNumber(1, "ZMin", 0.0)
# gmsh.model.mesh.field.setNumber(1, "ZMax", 0.1)
# gmsh.model.mesh.field.setNumber(1, "Thickness", 0.1)

# Let's use the minimum of all the fields as the mesh size field:
# gmsh.model.mesh.field.add("MinAniso", 2)
# gmsh.model.mesh.field.setNumbers(2, "FieldsList", [1])
# gmsh.model.mesh.field.setAsBackgroundMesh(2)


# gmsh.option.setNumber("Mesh.Algorithm",      5)  
# gmsh.option.setNumber("Mesh.Algorithm3D",      1)  
# gmsh.option.setNumber("Mesh.Recombine3DAll",   1)  # recombine tetra → hex
# gmsh.option.setNumber("Mesh.Recombine3DLevel",   0)  #
gmsh.option.setNumber("Mesh.RecombineAll",   1)  # recombine tri → quad
# gmsh.option.setNumber("Mesh.RecombinationAlgorithm",   3)  #
# gmsh.option.setNumber("Mesh.Recombine3DConformity",   0)  #
# gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)  # simple subdivision

gmsh.write(os.path.join(script_dir, "debug_output.geo_unrolled"))
# exit()

gmsh.model.mesh.generate(3)

# 4. write & show ---------------------------------------------
# Get the absolute directory where *this* Python script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build full paths
mesh_filename = os.path.join(script_dir, "naca4412_cgrid3D_quad.msh")
cgns_filename = os.path.join(script_dir, "naca4412_cgrid3D_quad.cgns")

# Then write
gmsh.write(mesh_filename)
gmsh.write(cgns_filename)
# if "-nopopup" not in sys.argv:
#     gmsh.fltk.run()
gmsh.finalize()
# --------------------------------------------------------------
