import gmsh
import sys
import numpy as np

def create_airfoil_cgrid_mesh(use_gui=True):
    """
    Creates a structured C-grid mesh around a NACA 0012 airfoil
    
    Parameters:
    -----------
    use_gui : bool
        If True, launches GMSH GUI for visualization
    """
    
    gmsh.initialize()
    gmsh.model.add("airfoil_cgrid")
    
    # ============================================
    # PARAMETERS
    # ============================================
    chord = 1.0
    n_airfoil = 81  # MUST BE ODD for symmetric airfoil (ensures TE point)
    n_wake = 21     # Number of points in wake
    n_radial = 41   # Radial divisions
    far_field_radius = 15.0 * chord
    wake_length = 5.0 * chord
    
    # ============================================
    # NACA 0012 AIRFOIL GENERATION
    # ============================================
    def naca0012(x):
        """NACA 0012 thickness distribution"""
        t = 0.12
        return 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 
                        0.2843*x**3 - 0.1015*x**4)
    
    # Generate airfoil coordinates
    # Upper surface: LE to TE (0 to 1)
    x_upper = np.linspace(0, 1, n_airfoil)
    y_upper = naca0012(x_upper)
    
    # Lower surface: TE to LE (1 to 0)
    x_lower = np.linspace(1, 0, n_airfoil)
    y_lower = -naca0012(x_lower)
    
    # ============================================
    # CREATE AIRFOIL POINTS AND CURVES
    # ============================================
    airfoil_points_upper = []
    airfoil_points_lower = []
    
    # Upper surface points (LE to TE)
    for i in range(n_airfoil):
        pt = gmsh.model.geo.addPoint(x_upper[i], y_upper[i], 0)
        airfoil_points_upper.append(pt)
    
    # Lower surface points (TE to LE) - use TE point from upper surface, then continue
    airfoil_points_lower = [airfoil_points_upper[-1]]  # Start with TE point from upper
    
    for i in range(1, n_airfoil):
        pt = gmsh.model.geo.addPoint(x_lower[i], y_lower[i], 0)
        airfoil_points_lower.append(pt)
    
    # Create splines for airfoil
    airfoil_upper = gmsh.model.geo.addSpline(airfoil_points_upper)
    airfoil_lower = gmsh.model.geo.addSpline(airfoil_points_lower)
    
    # ============================================
    # CREATE WAKE POINTS
    # ============================================
    wake_upper_pts = []
    wake_lower_pts = []
    
    # Wake extends from trailing edge
    x_wake = np.linspace(1.0, 1.0 + wake_length, n_wake)
    
    # Upper wake (small positive y)
    for i in range(n_wake):
        y_val = 0.001 * (1.0 - i/(n_wake-1))  # Gradually approach centerline
        pt = gmsh.model.geo.addPoint(x_wake[i], y_val, 0)
        wake_upper_pts.append(pt)
    
    # Lower wake (small negative y)
    for i in range(n_wake):
        y_val = -0.001 * (1.0 - i/(n_wake-1))
        pt = gmsh.model.geo.addPoint(x_wake[i], y_val, 0)
        wake_lower_pts.append(pt)
    
    # Create wake curves
    wake_upper = gmsh.model.geo.addSpline(wake_upper_pts)
    wake_lower = gmsh.model.geo.addSpline(wake_lower_pts)
    
    # ============================================
    # CREATE FAR-FIELD C-GRID BOUNDARY
    # ============================================
    
    # Calculate angles for C-grid - split into upper and lower halves
    theta_lower_start = -np.pi / 2
    theta_mid = 0.0
    theta_upper_end = np.pi / 2
    
    # Upper far-field arc (matches upper airfoil points)
    n_ff_upper = n_airfoil
    thetas_upper = np.linspace(theta_mid, theta_upper_end, n_ff_upper)
    
    farfield_upper_pts = []
    for theta in thetas_upper:
        x = -far_field_radius * np.cos(theta) + wake_length/2
        y = far_field_radius * np.sin(theta)
        pt = gmsh.model.geo.addPoint(x, y, 0)
        farfield_upper_pts.append(pt)
    
    # Lower far-field arc (matches lower airfoil points)
    n_ff_lower = n_airfoil
    thetas_lower = np.linspace(theta_mid, theta_lower_start, n_ff_lower)
    
    farfield_lower_pts = []
    for theta in thetas_lower:
        x = -far_field_radius * np.cos(theta) + wake_length/2
        y = far_field_radius * np.sin(theta)
        pt = gmsh.model.geo.addPoint(x, y, 0)
        farfield_lower_pts.append(pt)
    
    # Create upper and lower far-field curves
    farfield_upper_curve = gmsh.model.geo.addSpline(farfield_upper_pts)
    farfield_lower_curve = gmsh.model.geo.addSpline(farfield_lower_pts)
    
    # Far-field wake closure points
    farfield_wake_upper_end = gmsh.model.geo.addPoint(
        1.0 + wake_length, far_field_radius, 0
    )
    farfield_wake_lower_end = gmsh.model.geo.addPoint(
        1.0 + wake_length, -far_field_radius, 0
    )
    
    # Upper wake far-field line
    farfield_upper_wake = gmsh.model.geo.addLine(
        farfield_upper_pts[-1], farfield_wake_upper_end
    )
    
    # Lower wake far-field line
    farfield_lower_wake = gmsh.model.geo.addLine(
        farfield_lower_pts[-1], farfield_wake_lower_end
    )
    
    # Far-field wake outlet
    farfield_wake_outlet = gmsh.model.geo.addLine(
        farfield_wake_upper_end, farfield_wake_lower_end
    )
    
    # ============================================
    # CREATE RADIAL CONNECTING LINES
    # ============================================
    
    # Leading edge to far-field (connects upper and lower at LE)
    le_point = airfoil_points_upper[0]
    farfield_le_upper = farfield_upper_pts[0]
    farfield_le_lower = farfield_lower_pts[0]
    
    radial_le_upper = gmsh.model.geo.addLine(le_point, farfield_le_upper)
    radial_le_lower = gmsh.model.geo.addLine(le_point, farfield_le_lower)
    
    # Trailing edge upper to far-field
    te_upper_point = airfoil_points_upper[-1]
    radial_te_upper = gmsh.model.geo.addLine(
        te_upper_point, farfield_upper_pts[-1]
    )
    
    # Trailing edge lower to far-field
    te_lower_point = airfoil_points_lower[-1]
    radial_te_lower = gmsh.model.geo.addLine(
        te_lower_point, farfield_lower_pts[-1]
    )
    
    # Wake ends to far-field
    wake_upper_end = wake_upper_pts[-1]
    wake_lower_end = wake_lower_pts[-1]
    
    radial_wake_upper = gmsh.model.geo.addLine(
        wake_upper_end, farfield_wake_upper_end
    )
    radial_wake_lower = gmsh.model.geo.addLine(
        wake_lower_end, farfield_wake_lower_end
    )
    
    # ============================================
    # CREATE CURVE LOOPS AND SURFACES
    # ============================================
    
    # Block 1: Upper airfoil region (LE -> upper airfoil -> TE -> radial -> farfield -> radial -> LE)
    upper_loop = gmsh.model.geo.addCurveLoop([
        airfoil_upper,           # LE to TE along airfoil
        radial_te_upper,         # TE to far-field
        -farfield_upper_curve,   # Far-field back to LE (reversed)
        -radial_le_upper         # Far-field to LE (reversed)
    ])
    upper_surface = gmsh.model.geo.addPlaneSurface([upper_loop])
    
    # # Block 2: Lower airfoil region
    # lower_loop = gmsh.model.geo.addCurveLoop([
    #     airfoil_lower,           # LE to TE along airfoil (note: already goes TE to LE)
    #     radial_te_lower,         # TE to far-field
    #     -farfield_lower_curve,   # Far-field back to LE (reversed)
    #     -radial_le_lower         # Far-field to LE (reversed)
    # ])
    # lower_surface = gmsh.model.geo.addPlaneSurface([lower_loop])
    
    # # Block 3: Upper wake region
    # upper_wake_loop = gmsh.model.geo.addCurveLoop([
    #     wake_upper,              # TE to wake end
    #     radial_wake_upper,       # Wake end to far-field
    #     -farfield_upper_wake,    # Far-field back to TE far-field (reversed)
    #     -radial_te_upper         # Far-field to TE (reversed)
    # ])
    # upper_wake_surface = gmsh.model.geo.addPlaneSurface([upper_wake_loop])
    
    # # Block 4: Lower wake region
    # lower_wake_loop = gmsh.model.geo.addCurveLoop([
    #     wake_lower,              # TE to wake end
    #     radial_wake_lower,       # Wake end to far-field
    #     -farfield_lower_wake,    # Far-field back to TE far-field (reversed)
    #     -radial_te_lower         # Far-field to TE (reversed)
    # ])
    # lower_wake_surface = gmsh.model.geo.addPlaneSurface([lower_wake_loop])

    # ============================================
    # SYNCHRONIZE GEOMETRY
    # ============================================
    gmsh.model.geo.synchronize()

    # ============================================
    # SET TRANSFINITE MESHING
    # ============================================
    
    print("Setting transfinite curves...")
    
    # Airfoil curves
    gmsh.model.mesh.setTransfiniteCurve(airfoil_upper, n_airfoil)
    gmsh.model.mesh.setTransfiniteCurve(airfoil_lower, n_airfoil)
    
    # Wake curves
    gmsh.model.mesh.setTransfiniteCurve(wake_upper, n_wake)
    gmsh.model.mesh.setTransfiniteCurve(wake_lower, n_wake)
    
    # Radial curves (all should have same number of points)
    gmsh.model.mesh.setTransfiniteCurve(radial_le_upper, n_radial)
    gmsh.model.mesh.setTransfiniteCurve(radial_le_lower, n_radial)
    gmsh.model.mesh.setTransfiniteCurve(radial_te_upper, n_radial)
    gmsh.model.mesh.setTransfiniteCurve(radial_te_lower, n_radial)
    gmsh.model.mesh.setTransfiniteCurve(radial_wake_upper, n_radial)
    gmsh.model.mesh.setTransfiniteCurve(radial_wake_lower, n_radial)
    
    # Far-field curves (must match corresponding airfoil/wake curves)
    gmsh.model.mesh.setTransfiniteCurve(farfield_upper_curve, n_airfoil)
    gmsh.model.mesh.setTransfiniteCurve(farfield_lower_curve, n_airfoil)
    gmsh.model.mesh.setTransfiniteCurve(farfield_upper_wake, n_wake)
    gmsh.model.mesh.setTransfiniteCurve(farfield_lower_wake, n_wake)
    gmsh.model.mesh.setTransfiniteCurve(farfield_wake_outlet, n_radial)
    
    # ============================================
    # PHYSICAL GROUPS
    # ============================================
    
    # gmsh.model.addPhysicalGroup(1, [airfoil_upper, airfoil_lower], 
    #                             name="Airfoil")
    # gmsh.model.addPhysicalGroup(1, [farfield_upper_curve, farfield_lower_curve, 
    #                                 farfield_upper_wake, farfield_lower_wake, 
    #                                 farfield_wake_outlet], 
    #                             name="Farfield")
    # gmsh.model.addPhysicalGroup(2, [upper_surface, lower_surface, 
    #                                 upper_wake_surface, lower_wake_surface], 
    #                             name="Fluid")
    
    # ============================================
    # PREVIEW GEOMETRY BEFORE MESHING
    # ============================================
    
    print("\nOpening GUI to preview geometry...")
    print("Close the GUI window when ready to generate mesh.")
    # gmsh.fltk.run()
    
    # ============================================
    # GENERATE MESH
    # ===========================================
    
    # print("Generating 2D mesh...")
    gmsh.model.mesh.generate(2)
    
    # # ============================================
    # # SAVE AND DISPLAY
    # # ============================================
    
    # gmsh.write("airfoil_cgrid.msh")
    # gmsh.write("airfoil_cgrid.vtk")
    # print("Mesh saved to airfoil_cgrid.msh and airfoil_cgrid.vtk")
    
    # Print mesh statistics
    print("\n" + "="*50)
    print("MESH STATISTICS")
    print("="*50)
    print(f"Number of nodes: {len(gmsh.model.mesh.getNodes()[0])}")
    print(f"Number of elements: {len(gmsh.model.mesh.getElements()[1][0])}")
    print("="*50 + "\n")
    
    if use_gui:
        print("Launching GUI... Close the window to exit.")
        gmsh.fltk.run()
    
    gmsh.finalize()

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Check command line arguments
    use_gui = False
    
    if len(sys.argv) > 1:
        if sys.argv[1].lower() in ['gui', '--gui', '-g']:
            use_gui = True
    else:
        # Interactive prompt
        response = input("Launch GUI? (y/n): ").strip().lower()
        use_gui = (response == 'y' or response == 'yes')
    
    create_airfoil_cgrid_mesh(use_gui=use_gui)