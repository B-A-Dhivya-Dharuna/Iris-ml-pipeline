# test_iris.py
import os
import unittest
from iris import train_and_evaluate_model # Assuming your training function is named this

class TestIrisPipeline(unittest.TestCase):
    """
    Simple test to check if the model training step runs without error
    and saves the model artifact.
    """
    
    def test_model_training_success(self):
        # The main function handles data download and splitting internally.
        try:
            train_and_evaluate_model()
            # Check if the model file was created (inside the container during the workflow run)
            self.assertTrue(os.path.exists('models/iris_knn_model.pkl'))
        except Exception as e:
            self.fail(f"Model training failed with error: {e}")

if __name__ == '__main__':
    unittest.main()
