import csv
import os
import glob
import random
import numpy as np
from numpy import ma
import cv2

# Gets all the csv files in a folder and returns their filepaths
def get_csv_files_in_folder(folder_path):
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    return csv_files


# Reads the annotations and returns a numpy array for all annotations in a given csv file
def read_annotations_from_csv(file_path):
    annotations = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            frame_number, obj_id, x, y, w, h, obj_class, species, occluded, noisy_frame = [float(val) if val else 0 for val in row]
            annotation = (frame_number, obj_id, x, y, w, h, obj_class, species, occluded, noisy_frame)
            annotations.append(annotation)
    return np.asarray(annotations)


# Takes annotations and for a specific target object gets the measurements and adds a mask to the array with any frames
# that are not observed
def mask_measurements(annotations, target_obj_id=1, dim_obs=2):
    # Store Measurements in a Masked Array
    max_number_frames = int(np.max(annotations.T[0]) + 1)
    # Each measurement has shape 4, x,y,w,h
    shape = (max_number_frames, dim_obs)
    measurements = np.zeros(shape)
    mask = np.ones(shape, dtype=bool)
    measurements = np.ma.masked_array(measurements, mask)
    # Go through every annotation for a specific object id
    for frame_id, obj_id, min_x, min_y, width, height, obj_class, species, occluded, noisy_frame in annotations:
        if obj_id == target_obj_id:
            measurements[int(frame_id)] = [min_x, min_y]
    #measurements[100:150] = ma.masked
    print(f"Debug: Measurements for obj id 1 {measurements}")

    return measurements


# Loads all the images paths in a directory
def load_image_paths(directory):
    image_paths = []
    for filename in os.listdir(directory):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".bmp"):
            image_paths.append(os.path.join(directory, filename))
    return sorted(image_paths)



# Displays the annotated video with the original, filtered, and smoothed measurements for multiple objects
def display_annotated_video(image_paths, obj_ids, org_meas, filtered_meas, smoothed_meas, model="Velocity'"):
    for idx2, image_path in enumerate(image_paths):
        frame_delay = 10
        w = 20
        h = 20
        w_smoothed = 25
        h_smoothed = 25

        frame = cv2.imread(image_path)
        # Add legend
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1
        cv2.putText(frame, "Black: Original", (10, 30), font, font_scale, (0, 0, 0), font_thickness)
        cv2.putText(frame, "Blue: Filtered", (10, 45), font, font_scale, (255, 0, 0), font_thickness)
        cv2.putText(frame, "Red: Smoothed", (10, 60), font, font_scale, (0, 0, 255), font_thickness)
        cv2.putText(frame, f"Model: {model}", (10,75), font, font_scale, (255,255,255), font_thickness)
        for obj_id in obj_ids:
            try:
                filtered_x = filtered_meas[obj_id][idx2, 0]
                filtered_y = filtered_meas[obj_id][idx2, 1]
            except:
                filtered_x, filtered_y = (-2000,-2000)
                print("Filtered x/y not found. Setting (0,0). Length orgmeas = ", len(org_meas), "index=",idx2)
            try:
                smoothed_x = smoothed_meas[obj_id][idx2, 0]
                smoothed_y = smoothed_meas[obj_id][idx2, 1]
            except:
                smoothed_x, smoothed_y = (-2000,-2000)
                print("Smoothed x/y not found. Setting (0,0). Length orgmeas = ", len(org_meas), "index=",idx2)
            try:
                org_x, org_y = org_meas[obj_id][idx2]
            except:
                org_x, org_y = (-2000, -2000)
                print("Smoothed x/y not found. Setting (0,0). Length orgmeas = ", len(org_meas), "index=", idx2)
            #try:
                #org_w, org_h = org_meas[obj_id][idx2]
            # Draw bounding boxes for original, filtered, and smoothed positions
            if not (np.ma.is_masked(org_x) or np.ma.is_masked(org_y)):
                org_top_left = (int(org_x - w / 2), int(org_y - h / 2))
                org_bottom_right = (int(org_x + w / 2), int(org_y + h / 2))
                cv2.rectangle(frame, org_top_left, org_bottom_right, (0, 0, 0), 8)

            filtered_top_left = (int(filtered_x - w / 2), int(filtered_y - h / 2))
            filtered_bottom_right = (int(filtered_x + w / 2), int(filtered_y + h / 2))
            cv2.rectangle(frame, filtered_top_left, filtered_bottom_right, (255, 0, 0), 2)

            smoothed_top_left = (int(smoothed_x - w_smoothed / 2), int(smoothed_y - h_smoothed / 2))
            smoothed_bottom_right = (int(smoothed_x + w_smoothed / 2), int(smoothed_y + h_smoothed / 2))
            cv2.rectangle(frame, smoothed_top_left, smoothed_bottom_right, (0, 0, 255), 2)

        cv2.imshow("Images as Video", frame)
        if cv2.waitKey(frame_delay) & 0xFF == ord("q"):
            break