import unittest
import torch
from Final import *

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.image_path = "Microbes/"
        self.batch_size = 32
        self.train_data, self.test_data = preprocess_data(self.image_path, batch_size=self.batch_size)

    def test_preprocess_data(self):
        # First batch of training data
        images, labels = next(iter(self.train_data))
        # Check the shape is correct
        self.assertEqual(images.shape, (self.batch_size, 1, 224, 224), "Incorrect image batch shape")
        self.assertEqual(labels.shape, (self.batch_size,), "Incorrect label batch shape")
        # Check the types are correct
        self.assertEqual(images.dtype, torch.float32, "Image dtype should be float32")
        self.assertEqual(labels.dtype, torch.int64, "Label dtype should be int64")
        # Check the labels are correct
        class_names = self.train_data.dataset.dataset.classes
        self.assertIn("E. Coli", class_names)
        self.assertIn("Staphylococcus", class_names)
        self.assertEqual(len(class_names), 2, "There should be exactly 2 classes")

if __name__ == '__main__':
    unittest.main()
