import unittest
import cv2
from detect_image import *
class MyTestCase(unittest.TestCase):
    def test_detect_image_detects_car(self):
        file_path_imgs = r'C:\Users\bubba\PycharmProjects\MultiBandIRTracking\TRI_A\TRI_A1\TRI_A1.mp4'
        cap = cv2.VideoCapture(file_path_imgs)
        self.assertEqual(cap.isOpened(), True)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv2.imshow('Detected Objects', detect_image(frame))
                cv2.imshow('Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        #.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
