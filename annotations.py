from utility import *
from tabulate import tabulate


def get_objects(data, min_detections=60):
    max_distance = 70  # Max distance to consider two objects as the same
    max_frame_gap = 500  # Max number of frames an object can be missed
    next_id = 1
    position_weight = 1.0  # Weight of the position difference in the distance metric
    size_weight = 0.1  # Weight of the size difference in the distance metric

    # Run through the frames
    frame_numbers, counts = np.unique(data[:, 0], return_counts=True)
    num_frames = len(frame_numbers)

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
                size = obj[4:6]
                obj_type = obj[6]

                min_distance = float('inf')
                obj_id = -1

                for k in range(len(candidate_frames)):
                    candidate_frame = candidate_frames[k]
                    candidate_frame_data = data[data[:, 0] == candidate_frame]

                    # Find closest object in candidate frame with the same object type
                    candidates = candidate_frame_data[candidate_frame_data[:, 6] == obj_type]

                    if candidates.size > 0:
                        position_distances = np.linalg.norm(candidates[:, 2:4] - position, axis=1)
                        size_distances = np.linalg.norm(candidates[:, 4:6] - size, axis=1)

                        # Distance metric that considers both position and size differences
                        combined_distances = position_weight * position_distances + size_weight * size_distances

                        cur_min_distance = np.min(combined_distances)
                        cur_min_idx = np.argmin(combined_distances)

                        if cur_min_distance < max_distance and cur_min_distance < min_distance:
                            min_distance = cur_min_distance
                            obj_id = candidates[cur_min_idx, 1]

                if obj_id == -1:
                    # No matching object, assign a new ID
                    obj_id = next_id
                    next_id += 1

                current_frame_data[j, 1] = obj_id
        # Update data
        data[data[:, 0] == current_frame, :] = current_frame_data

    # Threshold to remove artifacts/detections that are only a few frames
    # Iterate through the unique IDs and their counts
    unique_ids, counts = np.unique(data[:, 1], return_counts=True)

    removed_detections = []
    for obj_id, count in zip(unique_ids, counts):
        # If the count is below the threshold, remove all rows corresponding to that object ID
        if count < min_detections:
            removed_rows = data[data[:, 1] == obj_id]
            removed_detections.extend(removed_rows)
            data = data[data[:, 1] != obj_id]

    # Reassign the removed detections to existing objects
    for detection in removed_detections:
        frame, _, x, y, width, height, obj_type, *other = detection
        position = np.array([x, y])
        size = np.array([width, height])

        candidate_frame_data = data[data[:, 0] == frame]

        # Find closest object in candidate frame with the same object type
        candidates = candidate_frame_data[candidate_frame_data[:, 6] == obj_type]

        if candidates.size > 0:
            distances = np.linalg.norm(candidates[:, 2:4] - position, axis=1)
            size_differences = np.linalg.norm(candidates[:, 4:6] - size, axis=1)
            combined_scores = distances + size_differences
            cur_min_distance = np.min(combined_scores)
            cur_min_idx = np.argmin(combined_scores)

            if cur_min_distance < max_distance:
                # Update the closest object ID and distance
                obj_id = candidates[cur_min_idx, 1]

                # Update the data array with the reassigned ID
                detection[1] = obj_id
                data = np.vstack([data, detection])

    # Display the data with assigned IDs as a table
    headers = ["Frame", "ID", "X", "Y", "Width", "Height", "Obj_Type", "N/A", "N/A", "Confidence"]
    print(tabulate(data, headers=headers))

    # Display the data with assigned IDs
    return data


if __name__ == "__main__":
    annotations_dir = r'TRI_A/detections'
    csv_files = get_csv_files_in_folder(annotations_dir)
    for csv_file in csv_files:
        data = read_annotations_from_csv(csv_file)
        print(get_objects(data))
        with open('datafile.csv', 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            for row in data:
                csv_writer.writerow(row)