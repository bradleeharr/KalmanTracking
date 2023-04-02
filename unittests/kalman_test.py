import unittest
from kalman_filter import *

class TestKalmanTracking(unittest.TestCase):

    def test_kalman_tracking(self):
        # Test data
        annotations = [
            (0, 1, 50, 50, 20, 20, 'object_class', 'species', 0, 0),
            (1, 1, 52, 53, 20, 20, 'object_class', 'species', 0, 0),
            (2, 1, 54, 56, 20, 20, 'object_class', 'species', 0, 0),
            (0, 2, 80, 80, 30, 30, 'object_class', 'species', 0, 0),
            (1, 2, 81, 82, 30, 30, 'object_class', 'species', 0, 0),
        ]

        # Run the kalman_tracking function
        results = kalman_tracking(annotations)

        # Check if the number of output rows is the same as the input
        self.assertEqual(len(results), len(annotations))

        # Check if the output frame ids and object ids are the same as input
        for i in range(len(results)):
            self.assertEqual(results[i][0], annotations[i][0])  # frame id
            self.assertEqual(results[i][1], annotations[i][1])  # object id

        # Check if the output object_class, species, occluded and noisy_frame are the same as input
        print(annotations[i])
        print(results[i])
        for i in range(len(results)):
            self.assertEqual(results[i][6], annotations[i][6])  # object_class
            self.assertEqual(results[i][7], annotations[i][7])  # species
            self.assertEqual(results[i][8], annotations[i][8])  # occluded
            self.assertEqual(results[i][9], annotations[i][9])  # noisy_frame

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
