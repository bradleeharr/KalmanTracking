import csv
import os
import glob

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
