import numpy as np
import matplotlib.pyplot as plt
from utility import *

from pykalman import KalmanFilter

#def init_kalman_filter(obj_id, initial_state, configuration='position'):
def init_kalman_filter(obj_id, min_x, min_y, width, height, configuration='position'):
    if configuration == 'position':
        initial_state = [min_x, min_y, width, height]
        transition_matrices = np.eye(4)
        dim = 4
    elif configuration == 'velocity':
        initial_state = [min_x, min_y, width, height, 0, 0]
        transition_matrices = np.array([[1, 0, 0, 0, 1, 0],
                                        [0, 1, 0, 0, 0, 1],
                                        [0, 0, 1, 0, 0, 0],
                                        [0, 0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 1, 0],
                                        [0, 0, 0, 0, 0, 1]])
        dim = 6
    elif configuration == 'acceleration':
        initial_state = [min_x, min_y, width, height, 0, 0, 0, 0]
        transition_matrices = np.array([[1, 0, 0, 0, 1, 0, 0.5, 0],
                                        [0, 1, 0, 0, 0, 1, 0, 0.5],
                                        [0, 0, 1, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 1, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 1, 0, 1, 0],
                                        [0, 0, 0, 0, 0, 1, 0, 1],
                                        [0, 0, 0, 0, 0, 0, 1, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 1]])
        dim = 8
    else:
        raise ValueError("Invalid Kalman Filter configuration. Needs to be 'position' or 'velocity'.")

    kf = KalmanFilter(transition_matrices=transition_matrices,
                      observation_matrices=np.eye(dim),
                      initial_state_mean=initial_state,
                      transition_covariance=0.1 * np.eye(dim),
                      observation_covariance=1 * np.eye(dim),
                      initial_state_covariance=1 * np.eye(dim))
    return kf, initial_state

def kalman_tracking(annotations):
    # Dictionary to store Kalman filters for each object
    kf_dict = {}
    results = []

    # Can configure to include position, velocity, or acceleration state vectors
    configuration = 'velocity'
    for frame_id, obj_id, min_x, min_y, width, height, obj_class, species, occluded, noisy_frame in annotations:
        if obj_id not in kf_dict:
            # Initialize a Kalman filter for a new object
            kf, initial_state = init_kalman_filter(obj_id, min_x, min_y, width, height, configuration=configuration)
            kf_dict[obj_id] = kf
        else:
            kf = kf_dict[obj_id]

        # Perform the Kalman filter update step
        current_measurement = np.array([min_x, min_y, width, height])
        if obj_id in kf_dict:
            mean, covar = kf.filter_update(kf_dict[obj_id].initial_state_mean,
                                           kf_dict[obj_id].initial_state_covariance,
                                           current_measurement)
            kf_dict[obj_id].initial_state_mean = mean
            kf_dict[obj_id].initial_state_covariance = covar
        #mean, covar = kf.filter_update(kf.filter)#kf.filter(np.array([current_measurement]))[0]

        # Store the estimated position
        results.append([frame_id, obj_id, *mean, obj_class, species, occluded, noisy_frame])

    return results

def main():
    annotations_dir = r'C:\Users\bubba\PycharmProjects\MultiBandIRTracking\TrainReal\annotations'
    csv_files = get_csv_files_in_folder(annotations_dir)
    for idx, csv_file in enumerate(csv_files):
        original_annotations = read_annotations_from_csv(csv_file)
        filtered_annotations = kalman_tracking(original_annotations)

        # Calculate the distance between the original and filtered positions
        distances = []
        for orig_ann, filt_ann in zip(original_annotations, filtered_annotations):
            orig_x, orig_y = orig_ann[2], orig_ann[3]
            filt_x, filt_y = filt_ann[2], filt_ann[3]
            distance = np.sqrt((orig_x - filt_x) ** 2 + (orig_y - filt_y) ** 2)
            distances.append(distance)
            plt.figure(idx)

        # Plot the distance over time
        frame_numbers = [ann[0] for ann in original_annotations]
        plt.plot(frame_numbers, distances, label='Distance (Expected vs Filtered)')
        plt.xlabel('Frame Number')
        plt.ylabel('Distance')
        plt.legend()
        plt.title(f'Distance between Actual and Filtered Positions for {os.path.basename(csv_file)}')

        # Show all plots at once
        plt.show()

if __name__ == '__main__':
    main()