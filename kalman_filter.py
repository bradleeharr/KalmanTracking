import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

from utility import *
from collections import defaultdict
from numpy import ma

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
        t_cov, o_cov, initial_cov = (0, 0, 0)
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
        t_cov, o_cov, initial_cov = (40, 500, 100)
    else:
        raise ValueError("Invalid Kalman Filter configuration. Needs to be 'position' or 'velocity'.")

    initial_state = np.zeros(dim)
    initial_state[:4] = [min_x, min_y, width, height]
    observation_matrix = np.eye(dim)[:4]

    kf = KalmanFilter(transition_matrices=transition_matrices,
                      observation_matrices=observation_matrix,
                      initial_state_mean=initial_state,
                      transition_covariance=40 * np.eye(dim),
                      observation_covariance=500 * np.eye(4),
                      initial_state_covariance=100 * np.eye(dim))

    return kf, initial_state

def kalman_tracking(annotations, configuration: str):
    # Store Measurements in a Masked Array
    max_number_frames = np.max(annotations.T[0])+1
    # Each measurement has shape 4, x,y,w,h
    shape = (max_number_frames, 2)
    measurements = np.zeros(shape)
    mask = np.ones(shape, dtype=bool)
    measurements = np.ma.masked_array(measurements, mask)
    # Go through every annotation for a specific object id
    target_obj_id = 1;
    for frame_id, obj_id, min_x, min_y, width, height, obj_class, species, occluded, noisy_frame in annotations:
        if obj_id == target_obj_id:
            measurements[frame_id] = [min_x, min_y]
    measurements[100:150] = ma.masked
    print(f"Debug: Measurements for obj id 1 {measurements}")
    transition_matrices = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])

    kf = KalmanFilter(transition_matrices=transition_matrices, n_dim_obs=2)
    kf = kf.em(measurements, n_iter=15)
    (filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
    print(f"Debug: Filtered State means for obj id 1 {filtered_state_means}")

    states_pred = kf.em(measurements).smooth(measurements)[0]
    print('fitted model: {0}'.format(kf))

    pl.figure(figsize=(8, 4))
    x = np.linspace(1, max_number_frames, max_number_frames)
    pl.subplot(2,1,1)
    obs_scatter = pl.scatter(x, measurements.T[0], marker='x', color='b',
                             label='observations')
    position_line = pl.plot(x, states_pred[:, 0],
                            linestyle='-', marker='.', color='r',
                            label='position est.', alpha=0.5)
    pl.legend(loc='lower right')
    pl.title('Kalman Filtered Position for Elephant 1')
    pl.xlim(xmin=0, xmax=x.max())
    #pl.ylim(ymin=measurements.T[0].max()-200, ymax=measurements.T[0].max()+10)
    pl.ylabel('Distance')
    pl.subplot(2,1,2)
    velocity_line = pl.plot(x, states_pred[:, 1],
                            linestyle='-', marker='.', color='g',
                            label='velocity est.', alpha=0.5)
    plt.ylim(ymin=200, ymax=400)
    pl.xlabel('Frame Number')
    pl.legend(loc='lower right')
    pl.title('Modeled Velocity for Elephant 1')

    pl.show()

"""def kalman_tracking(annotations, configuration: str):
    # Dictionary to store Kalman filters for each object
    kf_dict = {}
    results = []
    measurements_dict = {}
    # Can configure to include position, velocity, or acceleration state vectors
    for frame_id, obj_id, min_x, min_y, width, height, obj_class, species, occluded, noisy_frame in annotations:
        if obj_id not in measurements_dict:
            # Initialize a new measurement list for each new object
            measurements = [min_x, min_y, width, height]
            measurements_dict[obj_id] = measurements
        measurements_dict[obj_id].append(measurements)
        measurements = measurements_dict[obj_id]
        if obj_id not in kf_dict:
            # Initialize a Kalman filter for a new object
            kf, initial_state = init_kalman_filter(obj_id, min_x, min_y, width, height, configuration=configuration)
            kf = KalmanFilter(em_vars=['transition_covariances', 'observation_covariance'])
            kf_dict[obj_id] = (kf, initial_state)
        else:
            kf, initial_state = kf_dict[obj_id]

        # Perform the Kalman filter update step
        new_measurement = np.array([min_x, min_y, width, height])
        means, covariances = kf.filter(measurements)
        next_mean, next_covariance = kf.filter_update( means[-1], covariances[-1], new_measurement)


        # Store the estimated position
        results.append([frame_id, obj_id, *next_mean, obj_class, species, occluded, noisy_frame])

    return results"""


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


def distance(x, y):
    return np.sqrt(np.square(x) + np.square(y))


def main():
    noise_std_deviation = 10
    annotations_dir = r'C:\Users\bubba\PycharmProjects\MultiBandIRTracking\TrainReal\annotations'
    configuration = 'acceleration'

    csv_files = get_csv_files_in_folder(annotations_dir)
    for idx, csv_file in enumerate(csv_files):
        original_annotations = read_annotations_from_csv(csv_file)
        #noisy_annotations = noisify_data(original_annotations, noise_std=noise_std_deviation)
        filtered_annotations = kalman_tracking(original_annotations, configuration)

        #original_positions = extract_positions(original_annotations)
        #noisy_positions = extract_positions(noisy_annotations)
        #filtered_positions = extract_positions(filtered_annotations)

        #time_axis = list(range(len(next(iter(original_positions.values()))['x'])))

        """for obj_id in original_positions:
            plt.figure()
            plt.plot(time_axis, original_positions[obj_id]['x'], color='blue', label='Original X')
            # plt.plot(time_axis, original_positions[obj_id]['y'], color='blue', linestyle='--', label='Original Y')
            plt.plot(time_axis, noisy_positions[obj_id]['x'], color='green', label='Noisy X')
            # plt.plot(time_axis, noisy_positions[obj_id]['y'], color='green', linestyle='--',
            #         label='Noisy (Corrupted) Y')
            plt.plot(time_axis, filtered_positions[obj_id]['x'], color='red', label='Filtered X')
            # plt.plot(time_axis, filtered_positions[obj_id]['y'], color='red', linestyle='--', label='Filtered Y')
            plt.xlabel('Time (frame number)')
            plt.ylabel('Position')
            plt.legend()
            plt.title(f'Original, Noisy, and Filtered Results over Time for Object {obj_id}')
            plt.show()"""


if __name__ == '__main__':
    main()
