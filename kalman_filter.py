import numpy as np
from pykalman import KalmanFilter
import matplotlib.pyplot as plt


def kalman_tracking(annotations):
    # Dictionary to store Kalman filters for each object
    kf_dict = {}

    # Result list to store estimated positions
    results = []

    for frame_id, obj_id, min_x, min_y, width, height, obj_class, species, occluded, noisy_frame in annotations:
        if obj_id not in kf_dict:
            # Initialize a Kalman filter for a new object
            kf = KalmanFilter(transition_matrices=np.eye(4),
                              observation_matrices=np.eye(4),
                              initial_state_mean=[min_x, min_y, width, height],
                              transition_covariance=0.1 * np.eye(4),
                              observation_covariance=1 * np.eye(4),
                              initial_state_covariance=1 * np.eye(4))
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
    original_annotations = [
        (0, 1, 50, 50, 20, 20, 'object_class', 'species', 0, 0),
        (1, 1, 52, 53, 20, 20, 'object_class', 'species', 0, 0),
        (2, 1, 54, 56, 20, 20, 'object_class', 'species', 0, 0),
        (0, 2, 80, 80, 30, 30, 'object_class', 'species', 0, 0),
        (1, 2, 81, 82, 30, 30, 'object_class', 'species', 0, 0),
    ]

    filtered_annotations = kalman_tracking(original_annotations)

    # Extract the original and filtered x, y positions
    original_x = [ann[2] for ann in original_annotations]
    original_y = [ann[3] for ann in original_annotations]
    filtered_x = [ann[2] for ann in filtered_annotations]
    filtered_y = [ann[3] for ann in filtered_annotations]

    # Plot the expected (original) and actual (filtered) results
    plt.scatter(original_x, original_y, color='blue', label='Original (Expected)')
    plt.scatter(filtered_x, filtered_y, color='red', label='Filtered (Actual)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Expected vs Actual Results')
    plt.show()

if __name__ == '__main__':
    main()