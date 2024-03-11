from dataobjects.trussElement import TrussElement



# Example usage
truss = TrussElement(radius=0.05, length=2.0)
print(f"Area: {truss.area}")
print(f"Surface Moment of Inertia: {truss.surfaceInertia}")
print(f"Polar Moment of Inertia: {truss.polarInertia}")