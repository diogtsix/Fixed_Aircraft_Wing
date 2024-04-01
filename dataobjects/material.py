class Material(): 
    
    def __init__(self, id: int = 1 , elasticModulus: float = 69*1e9, poissonRatio: float = 0.35,
                 density: float = 2700, 
                 shearModulus: float = 25.5*1e9, 
                 name :str =  "Aluminum"):
        
        """
        Initialize a new Material instance with physical properties.
        
        Parameters:
        - elasticModulus: Elastic modulus of the material (default: 69e9 [Pascal]).
        - poissonRatio: Poisson's ratio of the material (default: 0.35).
        - density: Density of the material (default: 2700 [kg/m3]).
        - shearModulus: Shear modulus of the material (default: 25.5e9 [Pascal]).
        
        """
        self.id = id
        self.elasticModulus = elasticModulus
        self.poissonRatio = poissonRatio
        self.density = density
        self.shearModulus = shearModulus
        self.name = name 
        