import logging
from typing import List, Tuple

import cv2
import numpy as np


class ObjectTracker:
    def __init__(self, boxes: list[tuple[int, int, int, int]]):
        self.boxes = boxes
        self.tracker = cv2.legacy.MultiTracker_create()
        self.initialized = False

    def set_tracker(self, tracker):
        self.tracker = tracker

    def initialize_trackers(self, frame: np.ndarray) -> bool:
        """
        Initialize object trackers
        :param frame: Initial frame to perform tracking on
        :return: Returns if tracking is successful
        """
        if frame is None:
            logging.error("Failed to read the frame")
            return False

        for box in self.boxes:
            self.tracker.add(cv2.legacy.TrackerMOSSE_create(), frame, tuple(box)) # TODO: Apply and Benchmark OpenCV Trackers
        self.initialized = True
        return True

    def update_tracker(self, frame: np.ndarray) -> np.ndarray:
        if not self.initialized:
            logging.error("Tracker not initialized.")
            return frame

        success, boxes = self.tracker.update(frame)
        if success:
            for box in boxes:
                x, y, w, h = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame
