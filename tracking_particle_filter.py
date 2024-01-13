import numpy as np
from particle_filter import ParticleFilter
from utility import *
from plotting import *
from annotations_test import *


def particle_tracking(data, model='Velocity', plot=True):
    # Group annotations by object ID
    object_ids = np.unique(data[:, 1])
    annotations_by_id = {obj_id: data[data[:, 1] == obj_id] for obj_id in object_ids}

    # Create a dictionary of particle filters, one for each unique object ID
    n_particles = 100
    state_dim = 6 if model == 'Acceleration' else 4
    measurement_dim = 2

    particle_filters = {obj_id: ParticleFilter(n_particles, state_dim, measurement_dim) for obj_id in object_ids}

    # Iterate through the frames and update each particle filter with the corresponding object's annotations
    frame_numbers = np.unique(data[:, 0])
    num_frames = len(frame_numbers)

    for i in range(num_frames):
        current_frame = frame_numbers[i]

        for obj_id in object_ids:
            annotations = annotations_by_id[obj_id]
            current_frame_data = annotations[annotations[:, 0] == current_frame]

            if current_frame_data.size > 0:
                # Replace these with your motion model and likelihood function
                motion_model = lambda state: state
                noise_covariance = np.eye(state_dim)
                likelihood_function = lambda measurement, state: 1

                particle_filter = particle_filters[obj_id]
                particle_filter.predict(motion_model, noise_covariance)
                particle_filter.update(current_frame_data[0, 2:4], likelihood_function)
                particle_filter.resample()

                estimated_state = particle_filter.estimate()
                # Update the original data with the estimated state (if needed)
                data[(data[:, 0] == current_frame) & (data[:, 1] == obj_id), 2:6] = estimated_state

def main():
    mode = 'BIRDSAI'
    if mode == 'BIRDSAI':
        annotations_dir = r'TrainReal/annotations'
        tracking_function = particle_tracking
    elif mode == 'TRICLOBS':
        annotations_dir = r'TRI_A/detections'
        tracking_function = particle_tracking

    # annotation format frame_number,object_id,x,y,w,h,class,species,occlusion,noise
    csv_files = get_csv_files_in_folder(annotations_dir)
    imgdirs = []
    for idx, file in enumerate(csv_files):
        print("Found csv file :", os.path.basename(file))
        if mode == 'BIRDSAI':
            imgdirs.append(r'TrainReal/images/' + os.path.basename(file)[:-4])
        elif mode == 'TRICLOBS':
            filename = r'TRI_A/' + os.path.basename(file)[11:-4] + '/frames'
            imgdirs.append(filename)
            print(f'added file to imgdir {filename}')
    for idx, csv_file in enumerate(csv_files):
        original_annotations = read_annotations_from_csv(csv_file)
        # In TRICLOBS, perform the object identification algorithm
        if mode == 'TRICLOBS': original_annotations = get_objects(original_annotations);
        obj_ids = np.unique(original_annotations[:, 1])
        image_paths = load_image_paths(imgdirs[idx])
        for model in ['Velocity', 'Acceleration']:
            org_meas, filtered_meas, smoothed_meas = tracking_function(original_annotations, model, plot=False)
            print(org_meas)
            print("Display Video:")
            display_annotated_video(image_paths, obj_ids, org_meas, filtered_meas, smoothed_meas,
                                    model=model)

        # mse = np.mean((ground_truth - filtered_state_means) ** 2)

if __name__ == '__main__':
    main()
