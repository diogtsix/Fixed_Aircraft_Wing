import numpy as np
import gmsh
import sys
import os

# Ensure the gmsh module is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def generate_wing_mesh(nodesAirfoil, wing_length, file_path):
    # Initialize gmsh
    gmsh.initialize()
    gmsh.model.add("wing_surface")

    # Create points in Gmsh from nodesAirfoil
    point_tags = []
    for i, (x, y) in enumerate(nodesAirfoil):
        point_tags.append(gmsh.model.geo.addPoint(x, y, 0, 1.0, i + 1))

    # Create lines connecting these points to form the airfoil shape
    line_tags = []
    num_points = len(nodesAirfoil)
    for i in range(num_points - 1):
        line_tags.append(gmsh.model.geo.addLine(point_tags[i], point_tags[i + 1]))
    line_tags.append(gmsh.model.geo.addLine(point_tags[-1], point_tags[0]))  # Close the loop

    # Debug: Print points and lines
    print(f"Points: {point_tags}")
    print(f"Lines: {line_tags}")

    # Create a curve loop and a plane surface
    curve_loop = gmsh.model.geo.addCurveLoop(line_tags)
    surface = gmsh.model.geo.addPlaneSurface([curve_loop])

    # Define the number of intermediate sections
    num_sections = 10
    dz = wing_length / num_sections

    # Extrude the surface in the z-direction with intermediate sections
    extruded_surfaces = gmsh.model.geo.extrude([(2, surface)], 0, 0, wing_length, [num_sections])

    # Debug: Print extruded surfaces
    print("Extruded surfaces:", extruded_surfaces)

    # Synchronize to ensure the geometry is correct
    gmsh.model.geo.synchronize()

    # Generate 2D mesh
    gmsh.model.mesh.generate(2)

    # Generate 3D mesh
    gmsh.model.mesh.generate(3)

    # Save the mesh to the specified file
    try:
        gmsh.write(file_path)
        print(f"Mesh successfully saved to {file_path}")
    except Exception as e:
        print(f"Failed to save mesh: {e}")

    # Check if the file was saved correctly
    if os.path.exists(file_path):
        print(f"Mesh successfully saved to {file_path}")
    else:
        print(f"Failed to save mesh to {file_path}")

    # Visualize the mesh
    try:
        gmsh.fltk.run()
    except Exception as e:
        print(f"Error during visualization: {e}")

    # Print detailed information about elements
    element_types, element_tags, node_tags = gmsh.model.mesh.getElements()
    for i, elem_type in enumerate(element_types):
        print(f"Element type: {elem_type}")
        print(f"Number of elements of this type: {len(element_tags[i])}")

    # Finalize gmsh
    gmsh.finalize()

def load_and_visualize_mesh(file_path):
    # Initialize gmsh
    gmsh.initialize()
    gmsh.open(file_path)

    # Print general mesh information
    print("Mesh information:")
    print(f"Number of nodes: {gmsh.model.mesh.getNodes()[0].size}")
    print(f"Number of elements: {len(gmsh.model.mesh.getElements()[0])}")

    # Print detailed information about elements
    element_types, element_tags, node_tags = gmsh.model.mesh.getElements()
    for i, elem_type in enumerate(element_types):
        print(f"Element type: {elem_type}")
        print(f"Number of elements of this type: {len(element_tags[i])}")

    # Visualize the mesh
    try:
        gmsh.fltk.run()
    except Exception as e:
        print(f"Error during visualization: {e}")

    # Finalize gmsh
    gmsh.finalize()

def main():
    directory = "solvers/fluid_dynamics"
    file_path = os.path.join(directory, "wing_surface_mesh.msh")

    if os.path.exists(file_path):
        print(f"Mesh file {file_path} already exists. Loading and visualizing...")
        load_and_visualize_mesh(file_path)
    else:
        print(f"Mesh file {file_path} does not exist. Generating mesh...")
        nodesAirfoil = np.array([
            [1.000000, -0.000000],
            [0.917284,  0.018842],
            [0.834388,  0.035845],
            [0.751335,  0.051152],
            [0.668142,  0.064790],
            [0.584823,  0.076659],
            [0.501392,  0.086518],
            [0.417862,  0.093954],
            [0.334251,  0.098330],
            [0.250587,  0.098652],
            [0.166908,  0.093233],
            [0.079453,  0.077656],
            [0.000000,  0.000000],
            [0.087214, -0.038767],
            [0.166425, -0.053452],
            [0.249413, -0.059763],
            [0.332415, -0.061018],
            [0.415471, -0.058906],
            [0.498608, -0.054419],
            [0.581843, -0.048195],
            [0.665191, -0.040647],
            [0.748665, -0.032017],
            [0.832279, -0.022402],
            [0.916049, -0.011778],
            [1.000000,  0.000000]
        ])
        wing_length = 10.0  # Set the desired wing length
        generate_wing_mesh(nodesAirfoil, wing_length, file_path)

if __name__ == "__main__":
    main()
