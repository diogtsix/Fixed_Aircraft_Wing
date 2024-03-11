from dataobjects.trussElement import TrussElement
from dataobjects.naca_airfoil_generation    import naca2D


# Example usage
truss = TrussElement(radius=0.05, length=2.0)
print(f"Area: {truss.area}")
print(f"Surface Moment of Inertia: {truss.surfaceInertia}")
print(f"Polar Moment of Inertia: {truss.polarInertia}")

nodes_airfoil, elements_airfoil = naca2D(chordLength=1.12 , plotAirfoil = True)

print("Nodes Airfoil:\n", nodes_airfoil)
print("Elements Airfoil:\n", elements_airfoil)