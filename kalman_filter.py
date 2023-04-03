import numpy as np
import matplotlib.pyplot as plt
from utility import *
from collections import defaultdict

from pykalman import KalmanFilter


# def init_kalman_filter(obj_id, initial_state, configuration='position'):
def init_kalman_filter(obj_id, min_x, min_y, width, height, configuration='position'):
    if configuration == 'position':
        transition_matrices = np.eye(4)
        dim = 4
    elif configuration == 'velocity':
        transition_matrices = np.array([[1, 0, 0, 0, 1, 0],
                                        [0, 1, 0, 0, 0, 1],
                                        [0, 0, 1, 0, 0, 0],
                                        [0, 0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 1, 0],
                                        [0, 0, 0, 0, 0, 1]])
        dim = 6
    elif configuration == 'acceleration':
        transition_matrices = np.array([[1, 0, 0, 0, 1, 0, 0.5, 0],
                                        [0, 1, 0, 0, 0, 1, 0, 0.5],
                                        [0, 0, 1, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 1, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 1, 0, 1, 0],
                                        [0, 0, 0, 0, 0, 1, 0, 1],
                                        [0, 0, 0, 0, 0, 0, 1, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 1]])
        dim = 8
        t_cov, o_cov, initial_cov = (10, 100, 25)
    else:
        raise ValueError("Invalid Kalman Filter configuration. Needs to be 'position' or 'velocity'.")

    initial_state = np.zeros(dim)
    initial_state[:4] = [min_x, min_y, width, height]
    observation_matrix = np.eye(dim)[:4]

    kf = KalmanFilter(transition_matrices=transition_matrices,
                      observation_matrices=observation_matrix,
                      initial_state_mean=initial_state,
                      transition_covariance=10 * np.eye(dim),
                      observation_covariance=100 * np.eye(4),
                      initial_state_covariance=25 * np.eye(dim))
    return kf, initial_state


def kalman_tracking(annotations):
    # Dictionary to store Kalman filters for each object
    kf_dict = {}
    results = []

    # Can configure to include position, velocity, or acceleration state vectors
    configuration = 'acceleration'
    for frame_id, obj_id, min_x, min_y, width, height, obj_class, species, occluded, noisy_frame in annotations:
        if obj_id not in kf_dict:
            # Initialize a Kalman filter for a new object
            kf, initial_state = init_kalman_filter(obj_id, min_x, min_y, width, height, configuration=configuration)
            kf_dict[obj_id] = (kf, initial_state)
        else:
            kf, initial_state = kf_dict[obj_id]

        # Perform the Kalman filter update step
        current_measurement = np.array([min_x, min_y, width, height])
        mean, covar = kf.filter_update(initial_state, kf.initial_state_covariance, current_measurement)
        initial_state = mean
        # Store the estimated position
        results.append([frame_id, obj_id, *mean, obj_class, species, occluded, noisy_frame])

    return results

# Extract the original, noisy, and filtered x, y positions for each object
def extract_positions(annotations):
    positions = defaultdict(lambda: {'x': [], 'y': []})
    for ann in annotations:
        obj_id = ann[1]
        x = ann[2]
        y = ann[3]
        positions[obj_id]['x'].append(x)
        positions[obj_id]['y'].append(y)
    return positions

def distance(x,y):
    return np.sqrt(np.square(x) + np.square(y))

def main():
    annotations_dir = r'C:\Users\bubba\PycharmProjects\MultiBandIRTracking\TrainReal\annotations'
    csv_files = get_csv_files_in_folder(annotations_dir)
    for idx, csv_file in enumerate(csv_files):
        original_annotations = read_annotations_from_csv(csv_file)
        noisy_annotations = noisify_data(original_annotations)
        filtered_annotations = kalman_tracking(noisy_annotations)

        original_positions = extract_positions(original_annotations)
        noisy_positions = extract_positions(noisy_annotations)
        filtered_positions = extract_positions(filtered_annotations)

        time_axis = list(range(len(next(iter(original_positions.values()))['x'])))

        for obj_id in original_positions:
            plt.figure()
            plt.plot(time_axis, original_positions[obj_id]['x'], color='blue', label='Original X')
            plt.plot(time_axis, original_positions[obj_id]['y'], color='blue', linestyle='--', label='Original Y')
            plt.plot(time_axis, noisy_positions[obj_id]['x'], color='green', label='Noisy (Corrupted) X')
            plt.plot(time_axis, noisy_positions[obj_id]['y'], color='green', linestyle='--',
                     label='Noisy (Corrupted) Y')
            plt.plot(time_axis, filtered_positions[obj_id]['x'], color='red', label='Filtered X')
            plt.plot(time_axis, filtered_positions[obj_id]['y'], color='red', linestyle='--', label='Filtered Y')
            plt.xlabel('Time (frame number)')
            plt.ylabel('Position')
            plt.legend()
            plt.title(f'Original, Noisy, and Filtered Results over Time for Object {obj_id}')
            plt.show()


if __name__ == '__main__':
    main()
