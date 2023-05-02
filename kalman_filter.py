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


def kalman_tracking(annotations, configuration='velocity'):
    # Store Measurements in a Masked Array
    max_number_frames = int(np.max(annotations.T[0]) + 1)
    # Each measurement has shape 4, x,y,w,h
    shape = (max_number_frames, 2)
    measurements = np.zeros(shape)
    mask = np.ones(shape, dtype=bool)
    measurements = np.ma.masked_array(measurements, mask)
    # Go through every annotation for a specific object id
    print(annotations[0,1])
    target_obj_id = annotations[0,1]
    for frame_id, obj_id, min_x, min_y, width, height, obj_class, species, occluded, noisy_frame in annotations:
        if obj_id == target_obj_id:
            measurements[int(frame_id)] = [min_x, min_y]
    measurements[100:150] = ma.masked
    print(f"Debug: Measurements for obj id 1 {measurements}")
    Δt = 1
    a = 0.5 * Δt ** 2
    if configuration == 'velocity':
        transition_matrices = np.array([[1, 0, Δt, 0],
                                        [0, 1, 0, Δt],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]])
    elif configuration == 'acceleration':
        transition_matrices = np.array([[1, 0, Δt, 0, a, 0],
                                        [0, 1, 0, Δt, 0, a],
                                        [0, 0, 1, 0, Δt, 0],
                                        [0, 0, 0, 1, 0, Δt],
                                        [0, 0, 0, 0, 1, 0],
                                        [0, 0, 0, 0, 0, 1]])
    kf = KalmanFilter(transition_matrices=transition_matrices, n_dim_obs=2)
    kf = kf.em(measurements, n_iter=15)
    (filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
    print(f"Debug: Filtered State means for obj id 1 {filtered_state_means}")

    #states_pred = kf.em(measurements).smooth(measurements)[0]
    print('fitted model: {0}'.format(kf))

    pl.figure(figsize=(8, 4))
    x = np.linspace(1, max_number_frames, max_number_frames)
    pl.subplot(2, 1, 1)
    obs_scatter = pl.scatter(x, measurements.T[0], marker='x', color='b',
                             label='observations')
    position_line = pl.plot(x, filtered_state_means[:, 0],
                            linestyle='-', marker='.', color='r',
                            label='position est.', alpha=0.5)
    pl.legend(loc='lower right')
    pl.title('Kalman Filtered Position for Elephant 1')
    pl.xlim(xmin=0, xmax=x.max())
    # pl.ylim(ymin=measurements.T[0].max()-200, ymax=measurements.T[0].max()+10)
    pl.ylabel('Distance')
    pl.subplot(2, 1, 2)
    velocity_line = pl.plot(x, filtered_state_means[:, 1],
                            linestyle='-', marker='.', color='g',
                            label='velocity est.', alpha=0.5)
    plt.ylim(ymin=200, ymax=400)
    pl.xlabel('Frame Number')
    pl.legend(loc='lower right')
    pl.title('Modeled Velocity for Elephant 1')

    pl.show()


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


def main():
    annotations_dir = r'TrainReal/annotations'
    #annotations_dir = r'C:\Users\bubba\PycharmProjects\MultiBandIRTracking\TRI_A\detections'
    configuration = 'velocity'
    # annotation format frame_number,object_id,x,y,w,h,class,species,occlusion,noise
    csv_files = get_csv_files_in_folder(annotations_dir)
    for file in csv_files:
        print("Found csv file :", os.path.basename(file))
    for idx, csv_file in enumerate(csv_files):
        original_annotations = read_annotations_from_csv(csv_file)
        filtered_annotations = kalman_tracking(original_annotations, configuration)


if __name__ == '__main__':
    main()
