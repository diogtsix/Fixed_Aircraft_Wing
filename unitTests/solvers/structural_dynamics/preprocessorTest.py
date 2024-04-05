
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from solvers.structural_dynamics.preprocessor import Preprocessor
import unittest
import numpy as np

class TestPreprocessor(unittest.TestCase):

    def test_initialization(self):
        """Test initialization of the Preprocessor class."""
        preprocessor = Preprocessor(chordLength= 1.12, wing_length= 3.61)
        self.assertEqual(preprocessor.chordLength, 1.12)
        self.assertEqual(preprocessor.wing_length, 3.61)
        # Continue for other attributes...

    def test_createWingNodeMatrix(self):
        """Test the creation of the wing node matrix."""
        preprocessor = Preprocessor()
        nodeMatrix = preprocessor.createWingNodeMatrix()
        self.assertEqual(len(nodeMatrix), preprocessor.numberOfAirfoils * 12)
        # Add more assertions to validate the contents of nodeMatrix...

    def test_createAirfoilNodeMatrix(self):
        """Test the creation of the airfoil node matrix."""
        preprocessor = Preprocessor()
        airfoilNodeMatrix = preprocessor.createAirfoilNodeMatrix()
        self.assertEqual(len(airfoilNodeMatrix), 12)  # Assuming a fixed number of nodes per airfoil
        # Additional checks...

    def test_addForces(self):
        """Test force assignment."""
        preprocessor = Preprocessor()
        preprocessor.addForces()
        # Assuming forces are added to specific nodes, verify this:
        self.assertTrue(np.array_equal(preprocessor.nodeMatrix[4].force, np.array([0, -preprocessor.forceValue, 0, 0, 0, 0])))
        # Repeat for other nodes...

    def test_boundaryConditions(self):
        """Test the correct determination of boundary condition DOFs."""
        preprocessor = Preprocessor()
        dofsToDelete = preprocessor.boundaryConditions()
        expected_indices = np.concatenate([np.arange(j*6+1, j*6+1+6) for j in [3, 4, 9, 10]])
        self.assertTrue(np.array_equal(dofsToDelete, expected_indices))
        
if __name__ == '__main__':
    unittest.main()