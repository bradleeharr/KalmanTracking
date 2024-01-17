import cv2
import logging
from labeling_gui import LabelingGUI
from object_tracker import ObjectTracker


# Open Webcam
cap = cv2.VideoCapture(0)
success, first_frame = cap.read()

if success:
    labeling_gui = LabelingGUI()
    boxes = labeling_gui.run(first_frame)
    logging.info("Boxes generated: ", boxes)
    tracker = ObjectTracker(boxes)

    # IF trackers succesfully iniitalized
    if tracker.initialize_trackers(first_frame):
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = tracker.update_tracker(frame)
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        logging.error("Failed to initialize trackers")



else:
    logging.error("Failed to read the first frame")

cap.release()
cv2.destroyAllWindows()

