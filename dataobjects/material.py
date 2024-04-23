class Material(): 
    
    def __init__(self, id: int = 1 , elasticModulus: float = 200*1e9, poissonRatio: float = 0.3,
                 density: float = 7850, 
                 shearModulus: float = 77*1e9, 
                 name :str =  "Steel", 
                 C = None, m = None ):
        
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
        self.C = C
        self.m = m
        