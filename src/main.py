
from install import setup_dataset
from object_tracker import ObjectTracker
from labeling_gui import LabelingGUI
import os
import cv2

if __name__ == "__main__":
    # Dataset
    dataset_folder_path = 'data/TrainReal'
    dataset_zip_file_path = 'data/datasets'
    dataset_url = 'https://lilablobssc.blob.core.windows.net/conservationdrones/v01/conservation_drones_train_real.zip'
    setup_dataset(dataset_folder_path, dataset_zip_file_path, dataset_url)

    # Scenes from Dataset
    base_folder = 'data/TrainReal/TrainReal/images'
    for scene_folder in sorted(os.listdir(base_folder)):
        full_path = os.path.join(base_folder, scene_folder)

        # Labeling the first frame
        scene_frame_paths = sorted(os.listdir(full_path))
        first_frame_path = os.path.join(full_path, scene_frame_paths[0])
        first_frame = cv2.imread(first_frame_path)
        labeling_gui = LabelingGUI()
        boxes = labeling_gui.run(first_frame)

        print(type(first_frame))
        # Tracking based on updates from first frame
        tracker = ObjectTracker(boxes)
        tracker.initialize_trackers(first_frame)

        for frame_path in scene_frame_paths:
            full_frame_path = os.path.join(full_path, frame_path)
            frame = cv2.imread(full_frame_path)

            if frame is not None:
                frame = tracker.update_tracker(frame)
                cv2.imshow("Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print(f"Failed to read frame: {full_frame_path}")




