import numpy as np
import csv
from utility import *
from tabulate import tabulate




def get_objects(data):
    max_distance = 60  # Max distance to consider two objects as the same
    max_frame_gap = 60  # Max number of frames an object can be missed
    next_id = 1

    # Run through the frames
    frame_numbers, counts = np.unique(data[:, 0], return_counts=True)
    num_frames = len(frame_numbers)
    max_objects = max(counts)
    for i in range(num_frames):
        current_frame = frame_numbers[i]
        current_frame_data = data[data[:, 0] == current_frame]

        if i == 0:
            # Assign initial IDs
            num_objects = current_frame_data.shape[0]
            current_frame_data[:, 1] = np.arange(1, num_objects + 1)
            next_id = num_objects + 1
        else:
            # Find frames within the max_frame_gap
            candidate_frames = frame_numbers[max(0, i - max_frame_gap):i]

            for j in range(current_frame_data.shape[0]):
                obj = current_frame_data[j, :]
                position = obj[2:4]
                obj_type = obj[6]

                min_distance = float('inf')
                obj_id = -1

                for k in range(len(candidate_frames)):
                    candidate_frame = candidate_frames[k]
                    candidate_frame_data = data[data[:, 0] == candidate_frame]

                    # Find closest object in candidate frame with the same object type
                    candidates = candidate_frame_data[candidate_frame_data[:, 6] == obj_type]

                    if candidates.size > 0:
                        distances = np.linalg.norm(candidates[:, 2:4] - position, axis=1)
                        cur_min_distance = np.min(distances)
                        cur_min_idx = np.argmin(distances)

                        if cur_min_distance < max_distance and cur_min_distance < min_distance:
                            # Update the closest object ID and distance
                            min_distance = cur_min_distance
                            obj_id = candidates[cur_min_idx, 1]

                if obj_id == -1:
                    # No matching object, assign a new ID
                    obj_id = next_id
                    next_id += 1

                current_frame_data[j, 1] = obj_id

        # Update data
        data[data[:, 0] == current_frame, :] = current_frame_data

    # Display the data with assigned IDs as a table
    headers = ["Frame", "ID", "X", "Y", "Width", "Height", "Obj_Type", "N/A", "N/A", "Confidence"]
    print(tabulate(data, headers=headers))

    # Display the data with assigned IDs
    return data



annotations_dir = r'TRI_A/detections'
csv_files = get_csv_files_in_folder(annotations_dir)
for csv_file in csv_files:
    data = read_annotations_from_csv(csv_file)
    print(get_objects(data))
    with open('datafile.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in data:
            csv_writer.writerow(row)