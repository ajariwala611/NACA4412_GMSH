import gmsh
import math
import os

# === Parameters ===
chord = 1.0
R = 5.0 * chord
Lwake = 10.0 * chord
span = 0.1 * chord
lc_wall = 1e-3 * chord
lc_far = 0.5 * chord
angle_res = 40  # semicircle resolution


# === Initialize GMSH ===
gmsh.initialize()
gmsh.model.add("naca4412_cgrid_3D")
gmsh.option.setNumber("General.Terminal", 1)

# === STEP 1: Import STL surface ===
script_dir = os.path.dirname(os.path.abspath(__file__))
# Path to STL file (change this)
stl_filename = "NACA4412_s10_new.stl"
stil_filepath = os.path.join(script_dir, stl_filename)
gmsh.merge(stil_filepath)
gmsh.model.mesh.classifySurfaces(
    angle=40 * math.pi / 180.0, includeBoundary=True,
    forceParametrizablePatches=True, curveAngle=180 * math.pi / 180.0
)
gmsh.model.mesh.createGeometry()
gmsh.model.occ.synchronize()

# === STEP 2: Create semicircular inlet face ===
theta = [math.pi/2 + i * math.pi / angle_res for i in range(angle_res + 1)]
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

# === STEP 6: Boolean fragment with STL surface ===
gmsh.model.occ.fragment(gmsh.model.occ.getEntities(3), gmsh.model.occ.getEntities(2))
gmsh.model.occ.synchronize()

# === STEP 7: Mesh generation settings ===
gmsh.option.setNumber("Mesh.MeshSizeMin", lc_wall)
gmsh.option.setNumber("Mesh.MeshSizeMax", lc_far)
gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
gmsh.option.setNumber("Mesh.Optimize", 1)

gmsh.model.mesh.generate(3)

# === STEP 8: Save ===
# Build full paths
mesh_filename = os.path.join(script_dir, "naca4412_cgrid2D_quad.msh")
cgns_filename = os.path.join(script_dir, "naca4412_cgrid2D_quad.cgns")
gmsh.finalize()
