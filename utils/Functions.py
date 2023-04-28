import os, requests, cv2
from PIL import Image
import threading
from random import shuffle
import yaml, json
from math import ceil
from tqdm import tqdm
import shutil,json
from pathlib import Path
import numpy as np

names = list()
def copying_data(path_given,data,color,use_segments):
    global names
    try:
        path_given_images = os.path.join(path_given, 'images')
        os.mkdir(path_given_images)
    except FileExistsError:
        pass

    try:
        path_given_labels = os.path.join(path_given, 'labels')
        os.mkdir(path_given_labels)
    except FileExistsError:
        pass
    for annotation in tqdm(data, colour=color):
        try:
            image_path = annotation['Local Storage Path']
            text_name = Path(annotation['Local Storage Path']).stem + '.txt'
            text_path = os.path.join(path_given_labels, text_name)
            img = cv2.imread(image_path)
            width, height = img.shape[1], img.shape[0]
            shutil.copy(image_path, os.path.join(path_given_images, Path(image_path).name))
            if not use_segments:
                for label in annotation['Label']['objects']:
                    try:
                        top, left, h, w = label['bbox'].values()
                        # xywh = [left,top, left + w,top,left + w,top+h,left,top + h]
                        xywh = [(left + w / 2) / width, (top + h / 2) / height, w / width, h / height]
                    except KeyError:
                        pass
                    cls = label['value']  # class name
                    if cls not in names:
                        names.append(cls)

                    line = names.index(cls), * xywh  # YOLO format (class_index, xywh)
                    with open(text_path, 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
            else:
                for label in annotation['Label']['objects']:
                    try:
                        segemetation = np.array([list(i.values()) for i in label['polygon']])
                        segemetation = (segemetation / np.array([width,height])).reshape(-1).tolist()
                    except KeyError:
                        pass
                    cls = label['value']  # class name
                    if cls not in names:
                        names.append(cls)

                    line = names.index(cls), * segemetation  # YOLO format (class_index, xywh)
                    with open(text_path, 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
        except:
            pass
    return path_given_images


def Convert_Json_to_Yolo(Folder_path_to_store_file,JSON_File,images_file,file,use_segments):
    global names
    file1 = open(JSON_File)
    data = json.load(file1)
    file1.close()
    fdata = []
    for i in data:
        if i['Label']['objects'] != []:
            fdata.append(i)
    data = fdata

    folder_name = Folder_path_to_store_file
    directory_path = folder_name

    try:
        os.mkdir(folder_name)
    except FileExistsError:
        pass

    try:
        train = os.path.join(directory_path, 'train')
        os.mkdir(train)
    except FileExistsError:
        pass

    try:
        test = os.path.join(directory_path, 'test')
        os.mkdir(test)
    except FileExistsError:
        pass

    try:
        val = os.path.join(directory_path, 'val')
        os.mkdir(val)
    except FileExistsError:
        pass

    images_file = os.path.normpath(images_file)
    data_file = os.path.normpath(images_file + os.sep + os.pardir)
    for annotation in data:
        if os.path.exists(os.path.normpath(os.path.join(data_file,annotation['Local Storage Path']))):
            annotation['Local Storage Path'] = os.path.normpath(os.path.join(data_file,annotation['Local Storage Path']))
        for label in annotation['Label']['objects']:
            cls = label['value']  # class name
            if cls not in names:
                names.append(cls)
    print(names)
    shuffle(data)
    train_index = ceil(len(data)*0.8)
    test_val_index = ceil(len(data)*0.1)
    train_data = data[:train_index]
    test_data = data[train_index:train_index+test_val_index]
    val_data = data[train_index+test_val_index:]
    print(len(train_data),len(test_data),len(val_data))
    t1 = threading.Thread(target=copying_data,args =(train,train_data,"green",use_segments))
    t2 = threading.Thread(target=copying_data, args=(val, val_data,"yellow",use_segments))
    t3 = threading.Thread(target=copying_data, args=(test, test_data,"magenta",use_segments))
    t1.start()
    t2.start()
    t3.start()
    t1.join()
    t2.join()
    t3.join()
    if os.path.exists(os.path.join(directory_path, Path(file).name.replace('json', 'yaml'))):
        pass
    else:
        d = {'path': directory_path,
             'train': os.path.join(train,'images'),
             'val': os.path.join(val,'images'),
             'test': os.path.join(test,'images'),
             'nc': len(names),
             'names': names}  # dictionary

        with open(os.path.join(directory_path, Path(file).name.replace('json', 'yaml')), 'w') as f:
            yaml.dump(d, f, sort_keys=False)

# Convert_Json_to_Yolo(Folder_path_to_store_file=r'D:\temp2',
#                      JSON_File=r"D:\temp\json\Final.json",
#                      images_file=r"D:\temp\images",
#                      file=r"D:\temp\json\Final.json")