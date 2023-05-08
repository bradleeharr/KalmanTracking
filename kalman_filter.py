import cv2
import numpy as np
import pylab as pl
from math import sqrt

from utility import *
from plotting import *
from annotations_test import *

from pykalman import KalmanFilter
#%from filterpy.kalman import KalmanFilter
#from filterpy.kalman import ExtendedKalmanFilter
from scipy.optimize import linear_sum_assignment
import numpy as np
from scipy.optimize import linear_sum_assignment

def hx(x):
    """ compute measurement for slant range that
    would correspond to state x.
    """

    return (x[0] ** 2 + x[2] ** 2) ** 0.5


def HJacobian_at(x):
    """ compute Jacobian of H matrix at x """

    horiz_dist = x[0]
    altitude = x[2]
    denom = sqrt(horiz_dist ** 2 + altitude ** 2)
    return np.array([[horiz_dist / denom, 0., altitude / denom]])

"""
def extended_kalman_tracking(annotations, model='Velocity', plot=True):
    max_number_frames = int(np.max(annotations.T[0]) + 1)
    measurements = mask_measurements(annotations)

    ekf = ExtendedKalmanFilter(dim_x=4, dim_z=2)
    ekf.x = measurements[0]
    dt = 0.05
    ekf.F = np.eye(3) + np.array([[0, 1, 0],
                                  [0, 0, 0],
                                  [0, 0, 0]]) * dt
    for z in measurements[1:]:
        ekf.update(np.array[z], HJacobian_at, hx)
        ekf.predict()

    # transition_matrices=transition_matrices, n_dim_obs=2)
    ekf = ekf.em(measurements, n_iter=15)
    (filtered_state_means, filtered_state_covariances) = ekf.filter(measurements)
    print(f"Debug: Filtered State means for obj id 1 {filtered_state_means}")
    print('fitted model: {0}'.format(ekf))
    plot_means_and_smoothed_and_measurements(filtered_state_means, measurements, max_number_frames, model=model)

    pl.show()
"""

def get_transition_matrix(model):
    Δt = 1
    a = 0.5 * Δt ** 2
    if model == 'Velocity':
        transition_matrices = np.array([[1, 0, Δt, 0],
                                        [0, 1, 0, Δt],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]])
    elif model == 'Acceleration':
        transition_matrices = np.array([[1, 0, Δt, 0, a, 0],
                                        [0, 1, 0, Δt, 0, a],
                                        [0, 0, 1, 0, Δt, 0],
                                        [0, 0, 0, 1, 0, Δt],
                                        [0, 0, 0, 0, 1, 0],
                                        [0, 0, 0, 0, 0, 1]])
    else:
        print("Configuration should be 'Velocity' or 'Acceleration'")
    return transition_matrices


def kalman_tracking(annotations, model='Velocity', plot=True):
    max_number_frames = int(np.max(annotations.T[0]) + 1)
    obj_ids = np.unique(annotations[:, 1])
    transition_matrices = get_transition_matrix(model)
    measurements = {}
    kalman_filters = {}
    smoothed_state_means = {}
    filtered_state_means = {}
    for obj_id in obj_ids:
        measurements[obj_id] = mask_measurements(annotations, obj_id)
        print("SHAPE MEASUREMENTS",np.shape(measurements[obj_id]))
        kalman_filters[obj_id] = KalmanFilter(transition_matrices=transition_matrices, n_dim_obs=2).em(
            measurements[obj_id][0:50], n_iter=15)

        (filtered_state_means[obj_id], filtered_state_covariances) = kalman_filters[obj_id].filter(measurements[obj_id])
        print(f"Debug: Filtered State means for obj id {obj_id}: {filtered_state_means[obj_id]}")
        smoothed_state_means[obj_id] = kalman_filters[obj_id].em(measurements[obj_id]).smooth(measurements[obj_id])[0]
        print("Smoothed State Means:", smoothed_state_means[obj_id])
        print('fitted model: {0}'.format(kalman_filters[obj_id]))
        if plot: plot_means_and_smoothed_and_measurements(filtered_state_means[obj_id], smoothed_state_means[obj_id],
                                                          measurements[obj_id],
                                                          max_number_frames, model=model, obj_id=obj_id)
    return measurements, filtered_state_means, smoothed_state_means

# function to compute the cost matrix given measurements and predictions
def compute_cost_matrix(detections, predicted_positions):
    cost_matrix = np.zeros((len(detections), len(predicted_positions)))
    for i, detection in enumerate(detections):
        for j, prediction in enumerate(predicted_positions):
            cost_matrix[i, j] = np.linalg.norm(detection - prediction)
    return cost_matrix



def iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    X1, Y1, X2, Y2 = bbox2
    x_intersection = max(0, min(x2, X2) - max(x1, X1))
    y_intersection = max(0, min(y2, Y2) - max(y1, Y1))
    intersection_area = x_intersection * y_intersection
    bbox1_area = (x2 - x1) * (y2 - y1)
    bbox2_area = (X2 - X1) * (Y2 - Y1)
    union_area = bbox1_area + bbox2_area - intersection_area
    return intersection_area / union_area


def kalman_tracking_2(annotations, model='Velocity', n_init_meas=10, plot=True):
    max_number_frames = int(np.max(annotations.T[0]) + 1)
    transition_matrices = get_transition_matrix(model)
    measurements = {}
    kalman_filters = {}
    smoothed_state_means = {}
    filtered_state_means = {}
    n_dim_state = transition_matrices.shape[0]
    frames, max_objects = np.unique(annotations[:,0], return_counts=True)
    print(np.unique(annotations[:,0], return_counts=True))
    print("MAX OBJECTS: ", max_objects)

    measurements = annotations[:, 2:3]

    """num_objects = 0
    if not kalman_filters:
        kf = KalmanFilter(transition_matrices=transition_matrices, n_dim_obs=2, n_dim_state=n_dim_state).em(np.array(measurements), n_iter=15)
        obj_id = len(kalman_filters)
        kalman_filters[obj_id] = kf
        (filtered_state_means[obj_id], _) = kf.filter(np.array[measurements[obj_id])
        smoothed_state_means[obj_id] = kf.smooth(measurements[obj_id])
    else:
        predictions = np.array([kf.transition_matrices @ state[-1] for kf, state in filtered_state_means.items()])
        cost_matrix = compute_cost_matrix(current_measurements, predictions)
        assignments = linear_sum_assignment(cost_matrix)

        for i, j in assignments:
            obj_id = j
            meas = current_measurements[i]
            kf = kalman_filters[obj_id]
            measurements[obj_id].append(meas)
            filtered_mean, _ = kf.filter_update(filtered_state_means[obj_id][-1], observation=meas)
            filtered_state_means[obj_id] = np.vstack([filtered_state_means[obj_id], filtered_mean])
            smoothed_mean = kf.smooth(np.vstack([measurements[obj_id], meas]))[0][-1]
            smoothed_state_means[obj_id] = np.vstack([smoothed_state_means[obj_id], smoothed_mean])

            if plot:
                plot_means_and_smoothed_and_measurements(filtered_state_means[obj_id],
                                                         smoothed_state_means[obj_id],
                                                         np.array(measurements[obj_id]),
                                                         max_number_frames, model=model, obj_id=obj_id)

    return measurements, filtered_state_means, smoothed_state_means
    """


def main():
    mode = 'TRICLOBS'
    if mode == 'BIRDSAI':
        annotations_dir = r'TrainReal/annotations'
        tracking_function = kalman_tracking
    elif mode == 'TRICLOBS':
        annotations_dir = r'TRI_A/detections'
        tracking_function = kalman_tracking

    # annotation format frame_number,object_id,x,y,w,h,class,species,occlusion,noise
    csv_files = get_csv_files_in_folder(annotations_dir)
    imgdirs = []
    for idx, file in enumerate(csv_files):
        print("Found csv file :", os.path.basename(file))
        if mode == 'BIRDSAI': imgdirs.append(r'TrainReal/images/' + os.path.basename(file)[:-4])
        elif mode == 'TRICLOBS':
            filename = r'TRI_A/' + os.path.basename(file)[11:-4] + '/frames'
            imgdirs.append(filename)
            print(f'added file to imgdir {filename}')
    for idx, csv_file in enumerate(csv_files):
        original_annotations = read_annotations_from_csv(csv_file)
        original_annotations = get_objects(original_annotations);
        obj_ids = np.unique(original_annotations[:, 1])
        image_paths = load_image_paths(imgdirs[idx])
        for model in ['Velocity', 'Acceleration']:
            org_meas, filtered_meas, smoothed_meas = tracking_function(original_annotations, model, plot=False)
            print(org_meas)
            print("Display Video:")
            display_annotated_video(image_paths, obj_ids, org_meas, filtered_meas, smoothed_meas, model=model)

        # mse = np.mean((ground_truth - filtered_state_means) ** 2)


if __name__ == '__main__':
    main()
