import csv
import os
import glob
import random
import numpy as np


def read_annotations_from_csv(file_path):
    annotations = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            frame_number, obj_id, x, y, w, h, obj_class, species, occluded, noisy_frame = map(int, row)
            annotation = (frame_number, obj_id, x, y, w, h, obj_class, species, occluded, noisy_frame)
            annotations.append(annotation)
    return annotations


def get_csv_files_in_folder(folder_path):
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    return csv_files


# Noisify annotation data to test tracking probability
def noisify_data(annotations, noise_probability=0.1, noise_length=5, noise_type='gaussian'):
    noisy_annotations = []
    noise_counter = 0

    if noise_type == 'gaussian':
        for ann in annotations:
            noisy_ann = list(ann)
            noise = np.random.normal(0, 45, 4)
            noisy_ann[2:6] = np.add(noisy_ann[2:6], noise)
            noisy_annotations.append(tuple(noisy_ann))
    elif noise_type == 'nonlinear gaussian':
        for ann in annotations:
            if random.random() < noise_probability and noise_counter == 0:
                noise_counter = noise_length

            if noise_counter > 0:
                noisy_ann = list(ann)
                # Add random noise to x, y, width, and height
                noise = np.random.normal(0, 50, 4)
                noisy_ann[2:6] = np.add(noisy_ann[2:6], noise)
                noisy_annotations.append(tuple(noisy_ann))
                noise_counter -= 1
            else:
                noisy_annotations.append(ann)

    return noisy_annotations
