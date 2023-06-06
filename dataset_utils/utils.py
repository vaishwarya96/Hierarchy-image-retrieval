import numpy as np
import os


def get_id_map(file_path, inv=False):
    selected_classes = {}
    f = open(file_path, 'r')
    lines = f.readlines()
    del lines[0]
    for line in lines:
        line = line.strip()
        data_class, class_id = line.split(',')[0], int(line.split(',')[1])
        selected_classes[data_class] = class_id
    if inv:
        return {v: k for k, v in selected_classes.items()}
    else:
        return selected_classes


def get_dataset(dataset_folder, id_map):
    result = {}
    x_set = []
    y_set = []
    print("Retrieving dataset from:", dataset_folder)
    image_dirs = next(os.walk(dataset_folder))[1]
    n_classes = len(image_dirs)
 
    for i, data_class in enumerate(image_dirs):
        class_id = data_class
        path = os.path.join(dataset_folder, data_class)
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        for file in files:
            x_set.append(os.path.join(path, file))
            y_set.append(class_id)
            result.setdefault(class_id, []).append(os.path.join(path, file))
    y_id = []
    for i in range(len(y_set)):
        y_id.append(id_map[y_set[i]])

    return np.array(x_set), np.array(y_id), result
