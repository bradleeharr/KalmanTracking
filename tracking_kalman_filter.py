import os
from typing import Dict, List, Tuple

import numpy as np
from pykalman import KalmanFilter

from plotting import plot_kalman_filter_results
from utility import (
    display_annotated_video,
    get_csv_files_in_folder,
    load_image_paths,
    mask_measurements,
    read_annotations_from_csv,
)


def get_transition_matrix(model: str) -> np.ndarray:
    Δt = 1
    a = 0.5 * Δt**2
    if model == "Velocity":
        transition_matrices = np.array([[1, 0, Δt, 0], [0, 1, 0, Δt], [0, 0, 1, 0], [0, 0, 0, 1]])
    elif model == "Acceleration":
        transition_matrices = np.array(
            [
                [1, 0, Δt, 0, a, 0],
                [0, 1, 0, Δt, 0, a],
                [0, 0, 1, 0, Δt, 0],
                [0, 0, 0, 1, 0, Δt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
    else:
        print("Configuration should be 'Velocity' or 'Acceleration'")
    return transition_matrices


def kalman_tracking(
    annotations,
    model="Velocity",
    plot=True) -> tuple[dict[int, np.ma.MaskedArray], dict[int, np.ndarray], dict[int, np.ndarray]]:

    max_number_frames = int(np.max(annotations.T[0]) + 1)
    obj_ids = np.unique(annotations[:, 1])
    transition_matrices = get_transition_matrix(model)
    measurements = {}
    kalman_filters = {}
    smoothed_state_means = {}
    filtered_state_means = {}
    mses = {}
    filtered_mses = {}
    smoothed_mses = {}
    for obj_id in obj_ids:
        measurements[obj_id] = mask_measurements(annotations, obj_id)
        print("SHAPE MEASUREMENTS", np.shape(measurements[obj_id]))
        kalman_filters[obj_id] = KalmanFilter(
            transition_matrices=transition_matrices, n_dim_obs=2
        ).em(measurements[obj_id][0:50], n_iter=15)
        # measurements[obj_id].mask[150:200] = True
        (filtered_state_means[obj_id], filtered_state_covariances) = kalman_filters[obj_id].filter(
            measurements[obj_id]
        )
        print(f"Debug: Filtered State means for obj id {obj_id}: {filtered_state_means[obj_id]}")
        smoothed_state_means[obj_id] = (
            kalman_filters[obj_id].em(measurements[obj_id]).smooth(measurements[obj_id])[0]
        )
        print("Smoothed State Means:", smoothed_state_means[obj_id])
        print("fitted model: {0}".format(kalman_filters[obj_id]))
        if plot:
            plot_kalman_filter_results(
                filtered_state_means[obj_id],
                smoothed_state_means[obj_id],
                measurements[obj_id],
                max_number_frames,
                model=model,
                obj_id=obj_id,
            )
        known_indices = np.where(~measurements[obj_id].mask)[0]
        unknown_indices = np.where(measurements[obj_id].mask)[0]
        x_known = measurements[obj_id][known_indices, 0]
        y_known = measurements[obj_id][known_indices, 1]
        x_imputed = np.interp(unknown_indices, known_indices, x_known)
        y_imputed = np.interp(unknown_indices, known_indices, y_known)
        imputed_measurements = measurements.copy()
        imputed_measurements[obj_id][unknown_indices, 0] = x_imputed
        imputed_measurements[obj_id][unknown_indices, 1] = y_imputed
        # print(filtered_state_means[obj_id].shape)
        filtered_mses[obj_id] = np.mean(
            (imputed_measurements[obj_id] - filtered_state_means[obj_id][:, :2]) ** 2
        )
        smoothed_mses[obj_id] = np.mean(
            (imputed_measurements[obj_id] - smoothed_state_means[obj_id][:, :2]) ** 2
        )
        mses[obj_id] = np.mean((filtered_state_means[obj_id] - smoothed_state_means[obj_id]) ** 2)

    print("FILTERED MSES: ", filtered_mses)
    print("SMOOTHED MSES: ", smoothed_mses)
    print("DIFFERENCE FILTERED/SMOOTHED MSES: ", mses)
    return measurements, filtered_state_means, smoothed_state_means


def run_filter(mode, plotting=False):
    if mode == "BIRDSAI":
        annotations_dir = r"src/data/TrainReal/TrainReal/annotations"
    elif mode == "TRICLOBS":
        annotations_dir = r"TRI_A/detections"

    # annotation format frame_number,object_id,x,y,w,h,class,species,occlusion,noise
    csv_files = get_csv_files_in_folder(annotations_dir)
    imgdirs = []
    for idx, file in enumerate(csv_files):
        print("Found csv file :", os.path.basename(file))
        if mode == "BIRDSAI":
            imgdirs.append(r"src/data/TrainReal/TrainReal/images/" + os.path.basename(file)[:-4])
        elif mode == "TRICLOBS":
            filename = r"TRI_A/" + os.path.basename(file)[11:-4] + "/frames"
            imgdirs.append(filename)
            print(f"added file to imgdir {filename}")
    for idx, csv_file in enumerate(csv_files):
        original_annotations = read_annotations_from_csv(csv_file)
        # In TRICLOBS, perform the object identification algorithm
        #if mode == "TRICLOBS":
            #original_annotations = get_objects(original_annotations)
        obj_ids = np.unique(original_annotations[:, 1])
        image_paths = load_image_paths(imgdirs[idx])
        for model in ["Velocity", "Acceleration"]:
            org_meas, filtered_meas, smoothed_meas = kalman_tracking(
                original_annotations, model, plot=plotting
            )
            print(org_meas)
            print("Display Video:")
            display_annotated_video(
                image_paths,
                obj_ids,
                org_meas,
                filtered_meas,
                smoothed_meas,
                model=model,
            )

        # mse = np.mean((ground_truth - filtered_state_means) ** 2)


if __name__ == "__main__":
    dataset = "BIRDSAI"
    plotting = False
    run_filter(dataset, plotting)
