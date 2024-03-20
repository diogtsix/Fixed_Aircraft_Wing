

class BeamElement():
    """Represents a truss element in a structural system."""
    
    def __init__(self, material = None ,radius = None ,  length = 1, area = None, 
                 node1 = None, node2 = None , elementType = "beam"):
        
        """
        Initializes a new TrussElement instance with specified material, length, and cross-sectional area.
        
        Parameters:
        - material: An instance of the Material class representing the material of the truss element.
        - radius: The radius of the truss element in meters (optional if area is provided).
        - length: The length of the truss element in meters.
        - area: The cross-sectional area of the truss element in square meters (optional if radius is provided).
        - node1: First node of the truss element (instance of Node class, optional).
        - node2: Second node of the truss element (instance of Node class, optional).
        """
        
        self.material = material  # Instance of Material class
       
        self.length = length  # Length of the truss element in meters     
        self.node1 = node1 
        self.node2 = node2  
        self.elementType = elementType 
        
        if area is not None and radius is None:
            self.area = area
            self.radius = self.calculate_radius() if not radius else radius  # Optionally calculate radius from area
        elif radius is not None:
            self.radius = radius
            self.area = self.calculate_area()
        else:
            raise ValueError("Either radius or area must be specified.")
        
         
        self.surfaceInertia = self.calculate_surface_moment_of_inertia() #surface moment of inertia
        self.polarInertia = self.calculate_polar_moment_of_inertia() # polar moment of inertia    
        
    
    
    def calculate_area(self):
        """Calculates and returns the cross-sectional area of the truss element based on its radius."""
        return 3.14159 * (self.radius ** 2)  # Pi * r^2        
    
    def calculate_radius(self):  
        return (self.area / 3.14159) ** 0.5
    
    def calculate_surface_moment_of_inertia(self):
        return (3.14159 * self.radius**4) / 4 # PI r^4 /4
    
    def calculate_polar_moment_of_inertia(self):
        return (3.14159 * self.radius**4) / 4 # PI r^4 /4