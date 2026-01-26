import gmsh
import numpy as np
import math
import os

# ---------------------- Helper Functions ----------------------
def naca4412(x, closed_TE=True):
    m, p, t = 0.04, 0.4, 0.12
    yt = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 +
                  0.2843*x**3 - (0.1036 if closed_TE else 0.1015)*x**4)
    yc = np.where(x < p, m/p**2*(2*p*x - x**2),
                  m/(1-p)**2*((1-2*p)+2*p*x - x**2))
    dyc = np.where(x < p, 2*m/p**2*(p-x),
                   2*m/(1-p)**2*(p-x))
    th = np.arctan(dyc)
    xu = x - yt*np.sin(th)
    yu = yc + yt*np.cos(th)
    xl = x + yt*np.sin(th)
    yl = yc - yt*np.cos(th)
    return xu, yu, xl, yl

# cosine clustering utility
def cosine_spacing(N, start=0.0, end=1.0):
    beta = np.linspace(0, math.pi, N)
    cluster = 0.5*(1 - np.cos(beta))
    return start + (end - start)*cluster

# ---------------------- Parameters ----------------------
c = 1.0            # chord length
R = 5.0*c          # far-field radius
dom_L = 10.0*c    # wake extent in x
dom_H = 5.0*c     # upstream semi-circle radius
Lz = 0.1*c         # span thickness
Nu, Nl = 200, 100  # airfoil resolution
Ntheta = 100       # far-field semicircle resolution
lc_wall = 1e-3*c
lc_far = 0.5*c
BL_th = 0.02*c     # boundary layer total thickness
chord = 1.0
R = 5.0 * chord
Lwake = 10.0 * chord
span = 0.1 * chord
lc_wall = 1e-3 * chord
lc_far = 0.5 * chord

# ---------------------- Initialize Gmsh ----------------------
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)
gmsh.model.add("NACA4412_3D_Tet")

# ---------------------- Build 2D C-grid surface ----------------------
# Trailing edge point
pt_TE = gmsh.model.geo.addPoint(c, 0.0, 0.0, lc_wall)
# === STEP 2: Create semicircular inlet face ===
theta = [math.pi/2 + i * math.pi / Ntheta for i in range(Ntheta + 1)]
xy_points = [(R * math.cos(t), R * math.sin(t)) for t in theta]

# Points
p_center = gmsh.model.occ.addPoint(0, 0, 0, lc_far)
point_tags = [gmsh.model.occ.addPoint(x, y, 0, lc_far) for x, y in xy_points]

# Curves
circle_lines = []
for i in range(len(point_tags)-1):
    circle_lines.append(gmsh.model.occ.addLine(point_tags[i], point_tags[i+1]))
l_down = gmsh.model.occ.addLine(point_tags[-1], p_center)
l_up   = gmsh.model.occ.addLine(p_center, point_tags[0])

# Loop & Surface
curve_loop = gmsh.model.occ.addCurveLoop(circle_lines + [l_down, l_up])
semi_disk = gmsh.model.occ.addPlaneSurface([curve_loop])

# === STEP 3: Extrude semicircle in spanwise z-direction ===
ext = gmsh.model.occ.extrude([(2, semi_disk)], 0, 0, span)
volume_tags = [e[1] for e in ext if e[0] == 3]  # get 3D volume tag of semi-cylinder

# === STEP 4: Create wake box downstream ===
wake_box = gmsh.model.occ.addBox(chord, -R, 0, Lwake, 2*R, span)

# === STEP 5: Boolean fragment (semi-cylinder + box) ===
gmsh.model.occ.fragment([(3, wake_box)], [(3, volume_tags[0])])
gmsh.model.occ.synchronize()

# ---------------------- Create 3D volume ----------------------
# Extrude the 2D surface by span thickness Lz (geometry extrude only)
# but mesh as tetrahedra (recombine=False)
next_tags = fm.extrude([(2, surf)], 0, 0, Lz, [], recombine=False)
# find volume tag
vol_tag = next(tag for dim, tag in next_tags if dim == 3)

# ---------------------- Mesh Field Setup ----------------------
# 1) Boundary layer near airfoil
fld_bl = gmsh.model.mesh.field.add("BoundaryLayer")
gmsh.model.mesh.field.setNumbers(fld_bl, "CurvesList", [air_curve])
gmsh.model.mesh.field.setNumber(fld_bl, "hwall_n", lc_wall)
gmsh.model.mesh.field.setNumber(fld_bl, "thickness", BL_th)
gmsh.model.mesh.field.setNumber(fld_bl, "ratio", 1.2)
gmsh.model.mesh.field.setNumber(fld_bl, "NbLayers", 20)

# 2) Spanwise clustering: distance from mid-plane z=Lz/2
mid_pt = gmsh.model.geo.addPoint(0,0,Lz/2, lc_far)
fld_dist = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(fld_dist, "PointsList", [mid_pt])
fld_thr = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(fld_thr, "InField", fld_dist)
gmsh.model.mesh.field.setNumber(fld_thr, "SizeMin", lc_wall)
gmsh.model.mesh.field.setNumber(fld_thr, "SizeMax", lc_far)
gmsh.model.mesh.field.setNumber(fld_thr, "DistMin", 0.0)
gmsh.model.mesh.field.setNumber(fld_thr, "DistMax", Lz/2)

# 3) Combine fields: minimum of both
fld_min = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(fld_min, "FieldsList", [fld_bl, fld_thr])
gmsh.model.mesh.field.setAsBackgroundMesh(fld_min)

# ---------------------- Generate Tetrahedral Mesh ----------------------
# Use Delaunay tet mesher (Algorithm3D=1)
gmsh.option.setNumber("Mesh.Algorithm3D", 1)
gmsh.model.mesh.generate(3)

# ---------------------- Export ----------------------
gmsh.write("naca4412_tet3D.msh")
gmsh.write("naca4412_tet3D.cgns")

gmsh.finalize()
