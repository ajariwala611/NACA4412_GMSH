import gmsh
import sys
import numpy as np
import math

# Initialize gmsh
gmsh.initialize()
gmsh.model.add("naca4412_pipe_extrusion")

# Use OpenCASCADE kernel
gmsh.model.occ.synchronize()

# NACA 4412 airfoil parameters
m = 0.04  # maximum camber (4%)
p = 0.4   # position of maximum camber (40% chord)
t = 0.12  # maximum thickness (12%)
c = 1.0
Nu = 100
Nl = 100
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
pt_TE = gmsh.model.occ.addPoint(c, 0.0, 0.0)  # x=1
pt_LE = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)  # x=0

# Build upper surface points from TE to LE (exclude TE, LE)
upper_pts = [pt_TE] + [gmsh.model.occ.addPoint(x*c, y*c, 0)
                       for x, y in zip(xu[1:-1], yu[1:-1])] + [pt_LE]

# # Build lower surface points from LE to TE (exclude LE, TE)
lower_pts = [pt_LE] + [gmsh.model.occ.addPoint(x*c, y*c, 0)
                       for x, y in zip(xl[1:-1], yl[1:-1])] + [pt_TE]

# Create splines
curve_upper = gmsh.model.occ.addSpline(upper_pts)
curve_lower = gmsh.model.occ.addSpline(lower_pts)

# Create closed wire
# airfoil_loop = gmsh.model.occ.addWire([curve_upper, curve_lower], checkClosed=True)
airfoil_loop = gmsh.model.occ.addWire([curve_upper])
gmsh.model.occ.synchronize()
gmsh.model.mesh.setTransfiniteCurve(airfoil_loop, 501)  # 101 nodes = 100 elements

# Create the square profile in YZ plane at origin
square_size = 0.1
# Define corner points of the square in YZ plane (X=0)
# Create square in a plane perpendicular to X-axis at the TE
# Since we're following the upper surface, create square in YZ plane
# start_x, start_y, start_z = airfoil_coords[0]
# p1 = gmsh.model.occ.addPoint(0, 0, 0) 
# p2 = gmsh.model.occ.addPoint(-square_size, 0, 0) 
# p3 = gmsh.model.occ.addPoint(-square_size, 0, square_size) 
# p4 = gmsh.model.occ.addPoint(0, 0, square_size)

# p1 = gmsh.model.occ.addPoint(0, 0, 0) 
# p2 = gmsh.model.occ.addPoint(0, square_size, 0) 
# p3 = gmsh.model.occ.addPoint(0, square_size, square_size) 
# p4 = gmsh.model.occ.addPoint(0, 0, square_size)

p1 = gmsh.model.occ.addPoint(c, 0, 0) 
p2 = gmsh.model.occ.addPoint(c, square_size, 0) 
p3 = gmsh.model.occ.addPoint(c, square_size, square_size) 
p4 = gmsh.model.occ.addPoint(c, 0, square_size)
# Create lines forming the square
l1 = gmsh.model.occ.addLine(p1, p2)
l2 = gmsh.model.occ.addLine(p2, p3)
l3 = gmsh.model.occ.addLine(p3, p4)
l4 = gmsh.model.occ.addLine(p4, p1)

# Create curve loop and surface
curve_loop = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
square_surface = gmsh.model.occ.addPlaneSurface([curve_loop])

# Synchronize before pipe operation
gmsh.model.occ.synchronize()

# Create transfinite curves for the square edges (4 elements each)
for line in [l1, l2, l3, l4]:
    gmsh.model.mesh.setTransfiniteCurve(line, 5)  # 5 nodes = 4 elements

# Set transfinite surface for the square
gmsh.model.mesh.setTransfiniteSurface(square_surface)
gmsh.model.mesh.setRecombine(2, square_surface,90)

# Create pipe extrusion: extrude square along airfoil spline
pipe_dimtags = gmsh.model.occ.addPipe([(2, square_surface)], airfoil_loop)
gmsh.model.occ.synchronize()

all_surfaces = gmsh.model.getEntities(2)

for dim,tag in all_surfaces:
    lines = gmsh.model.occ.getCurveLoops(tag)
    curve_tags = lines[1][0]  # Get first array from the list
    for line in curve_tags:
        gmsh.model.mesh.setTransfiniteCurve(line, 50)  # 5 nodes = 4 elements
    gmsh.model.mesh.setTransfiniteSurface(tag)
    gmsh.model.mesh.setRecombine(2, tag, 90)

# Set transfinite volume for structured mesh
volumes = gmsh.model.getEntities(3)

if volumes:
    for vol in volumes:
        gmsh.model.mesh.setTransfiniteVolume(vol[1])
        gmsh.model.mesh.setRecombine(3, vol[1],90)

# Generate 3D mesh
gmsh.model.mesh.generate(2)



# Save mesh
# gmsh.write("naca4412_pipe_extrusion.msh")

# Launch GUI (optional - comment out if running in batch mode)
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

# Finalize
gmsh.finalize()