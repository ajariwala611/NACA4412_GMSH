# ----------------- naca4412_block3D.py ------------------
import gmsh
import numpy as np
import math
import os
import sys

# ---------- editable parameters ------------------------
c = 1.0  # chord length
Lz = 0.1 * c  # span length
R_far = 5.0 * c  # upstream semi-circle radius
Lwake = 10.0 * c  # wake length
Nx = 50  # points along chord (X)
Ny = 20  # wall-normal points (Y)
Nz = 10  # spanwise points (Z)
lc_wall = 1.0e-3 * c  # characteristic length near wall
hwall_n = 1.0e-3 * c  # first wall-normal spacing
BL_thickness = 0.02 * c  # total boundary layer thickness
wall_ratio = 1.2  # wall-normal growth ratio
span_ratio = 1.05  # spanwise growth ratio
# --------------------------------------------------------

gmsh.initialize()
gmsh.model.add("NACA4412_3D_Block")


# helper functions
def naca4412(x, closed_TE=True):
    m, p, t = 0.04, 0.4, 0.12
    yt = (
        5
        * t
        * (
            0.2969 * np.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - (0.1036 if closed_TE else 0.1015) * x**4
        )
    )
    yc = np.where(
        x < p,
        m / p**2 * (2 * p * x - x**2),
        m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2),
    )
    dyc = np.where(x < p, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    th = np.arctan(dyc)
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    return xu, yu, xl, yl


def cosine_spacing(N, start=0, end=1):
    beta = np.linspace(0, math.pi, N)
    x = 0.5 * (1 - np.cos(beta))
    return start + (end - start) * x


# 1. Build airfoil points along chord
x_cos = cosine_spacing(Nx + 1, 0.0, 1.0)
xu, yu, xl, yl = naca4412(x_cos)


# 2. Wall-normal spacing (Y direction)
def generate_wall_normal(N, hwall, thickness, ratio):
    dy = [hwall]
    for _ in range(N - 1):
        dy.append(dy[-1] * ratio)
    dy = np.array(dy)
    dy = thickness * dy / dy.sum()
    y_wall = np.concatenate(([0.0], np.cumsum(dy)))
    return y_wall


y_wall = generate_wall_normal(Ny, hwall_n, BL_thickness, wall_ratio)


# 3. Spanwise spacing (Z direction)
def generate_spanwise(N, Lz, ratio):
    dz = [1.0]
    for _ in range(N - 1):
        dz.append(dz[-1] * ratio)
    dz = np.array(dz)
    dz = Lz * dz / dz.sum()
    z_span = np.concatenate(([0.0], np.cumsum(dz)))
    return z_span


z_span = generate_spanwise(Nz, Lz, span_ratio)

# 4. Create 3D points
pts = []
for k in range(Nz + 1):
    for j in range(Ny + 1):
        for i in range(Nx + 1):
            if j == 0:
                # On airfoil surface
                if i <= Nx // 2:
                    x = xu[i]
                    y = yu[i]
                else:
                    x = xl[i]
                    y = yl[i]
                pts.append(gmsh.model.occ.addPoint(x * c, y * c, z_span[k], lc_wall))
            else:
                # Off-wall points
                x = x_cos[i]
                pts.append(
                    gmsh.model.occ.addPoint(x * c, y_wall[j], z_span[k], lc_wall)
                )


# helper to access point IDs
def idx(i, j, k):
    return i + (Nx + 1) * j + (Nx + 1) * (Ny + 1) * k


# 5. Create hexahedral blocks
volumes = []
transfinite_curves = []
transfinite_surfaces = []
recombine_surfaces = []
transfinite_volumes = []
recombine_volumes = []

for k in range(Nz):
    for j in range(Ny):
        for i in range(Nx):
            corner_pts = [
                pts[idx(i, j, k)],
                pts[idx(i + 1, j, k)],
                pts[idx(i + 1, j + 1, k)],
                pts[idx(i, j + 1, k)],
                pts[idx(i, j, k + 1)],
                pts[idx(i + 1, j, k + 1)],
                pts[idx(i + 1, j + 1, k + 1)],
                pts[idx(i, j + 1, k + 1)],
            ]

            lns = []
            # 1. Bottom face edges
            for n in range(4):
                l = gmsh.model.occ.addLine(corner_pts[n], corner_pts[(n + 1) % 4])
                lns.append(l)
                transfinite_curves.append((l, Nx + 1))  # X-direction

            # 2. Vertical edges (wall-normal, Y)
            for n in range(4):
                l = gmsh.model.occ.addLine(corner_pts[n], corner_pts[n + 4])
                lns.append(l)
                transfinite_curves.append((l, Ny + 1))  # Y-direction

            # 3. Top face edges (z=z+1)
            for n in range(4):
                l = gmsh.model.occ.addLine(
                    corner_pts[4 + n], corner_pts[4 + (n + 1) % 4]
                )
                lns.append(l)
                transfinite_curves.append((l, Nx + 1))  # X-direction again

            # Now create surfaces
            # Bottom surface
            loop_bottom = gmsh.model.occ.addWire([lns[0], lns[1], lns[2], lns[3]])
            surf_bottom = gmsh.model.occ.addPlaneSurface([loop_bottom])
            transfinite_surfaces.append(surf_bottom)
            recombine_surfaces.append(surf_bottom)

            # Top surface
            loop_top = gmsh.model.occ.addWire([lns[8], lns[9], lns[10], lns[11]])
            surf_top = gmsh.model.occ.addPlaneSurface([loop_top])
            transfinite_surfaces.append(surf_top)
            recombine_surfaces.append(surf_top)

            # Side surfaces
            surf_sides = []
            for side in [(0, 4, 8, 5), (1, 5, 9, 6), (2, 6, 10, 7), (3, 7, 11, 4)]:
                loop = gmsh.model.occ.addWire(
                    [lns[side[0]], lns[side[1]], -lns[side[2]], -lns[side[3]]]
                )
                surf = gmsh.model.occ.addPlaneSurface([loop])
                transfinite_surfaces.append(surf)
                recombine_surfaces.append(surf)
                surf_sides.append(surf)

            # Build volume
            surface_loop = gmsh.model.occ.addSurfaceLoop(
                [surf_bottom] + surf_sides + [surf_top]
            )
            vol = gmsh.model.occ.addVolume([surface_loop])
            # print(f"Volume {vol} created from surface loop: {surf_bottom}, {surf_top}, {surf_sides}")
            # print(f"  total surfaces in this volume: {1 + 1 + len(surf_sides)}")
            transfinite_volumes.append(vol)
            recombine_volumes.append(vol)
            volumes.append(vol)

exit()
gmsh.model.occ.synchronize()

for line_id, npts in transfinite_curves:
    gmsh.model.mesh.setTransfiniteCurve(line_id, npts)

for surf_id in transfinite_surfaces:
    gmsh.model.mesh.setTransfiniteSurface(surf_id)

for surf_id in recombine_surfaces:
    gmsh.model.mesh.setRecombine(2, surf_id)

for vol_id in transfinite_volumes:
    try:
        gmsh.model.mesh.setTransfiniteVolume(vol_id)
    except Exception as e:
        print(f"Skipping volume {vol_id} due to error: {e}")


for vol_id in recombine_volumes:
    gmsh.model.mesh.setRecombine(3, vol_id)

gmsh.model.mesh.removeDuplicateNodes()

# 6. mesh settings
# gmsh.option.setNumber("Mesh.RandomizeMesh", 0)
# gmsh.option.setNumber("Mesh.RecombineAll", 1)
# gmsh.option.setNumber("Mesh.Algorithm", 8) # Quad-delaunay recombination
# gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)

gmsh.option.setNumber("General.NumThreads", os.cpu_count())
gmsh.model.mesh.generate(3)

# 7. Export
out_dir = os.path.dirname(os.path.abspath(__file__))
# gmsh.option.setNumber("Mesh.CGNSLibraryVersion", 3.3)
gmsh.write(os.path.join(out_dir, "naca4412_cgrid3D_hex.msh"))
gmsh.write(os.path.join(out_dir, "naca4412_cgrid3D_hex.cgns"))


gmsh.finalize()
# -----------------------------------------------------------
