import logging
import cv2


class LabelingGUI:
    def __init__(self):
        self.boxes = []
        self.drawing = False
        self.start_point = None
        self.end_point = None

    def draw_rectangle(self, event, x, y, flags, param):
        """
        Callback to handle mouse clicks to draw rectangle and add rectangle as bounding boxes for tracking.
        :param event: OpenCV Mouse Event Callback
        :param x: Mouse X position
        :param y: Mouse Y position
        :param flags: OpenCV event flags (unused)
        :param param:  OpenCV event param (unused)
        :return: None
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = self.start_point  # Update the end_point here
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1 = min(self.start_point[0], self.end_point[0])
            y1 = min(self.start_point[1], self.end_point[1])
            x2 = max(self.start_point[0], self.end_point[0])
            y2 = max(self.start_point[1], self.end_point[1])
            self.boxes.append((x1, y1, x2 - x1, y2 - y1))

    def run(self, frame):
        """
        Run the labeling Window
        :param frame: Frame to label objects and
        :return: boxes. An list of tuples [(x1,y1,x2,y2)]
        """
        if frame is None:
            logging.info("Failed to read the first frame")
            return []

        cv2.namedWindow("Frame")
        cv2.setMouseCallback("Frame", self.draw_rectangle)

        while True:
            temp_frame = frame.copy()
            if self.drawing and self.start_point and self.end_point:
                cv2.rectangle(temp_frame, self.start_point, self.end_point, (0, 255, 0), 2)
            for x, y, w, h in self.boxes:
                cv2.rectangle(temp_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("Frame", temp_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cv2.destroyAllWindows()
        return self.boxes
