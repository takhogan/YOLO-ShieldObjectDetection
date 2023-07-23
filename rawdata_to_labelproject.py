from ultralytics import YOLO
import cv2
import glob
import os
import json
import labelbox as lb
import labelbox.types as lb_types
import uuid
import sys
from utils import *

def video_to_image_dir(video_path, image_dir='datasets/images'):
    extract_frames(video_path, image_dir)

def image_dir_to_labels(dir_name, glob_string=None):
    pass

def image_dir_to_labelproject(dir_name, glob_string=None):
    pass

def label_dir_to_labelproject(dir_name, glob_string):
    pass

def reverse_coord_convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    # Reverse the multiplication by dw and dh
    cx = box[0] / dw  # center x-coordinate
    cy = box[1] / dh  # center y-coordinate
    w = box[2] / dw  # width
    h = box[3] / dh  # height

    # Calculate the top-left and bottom-right coordinates from the center and dimensions
    x = cx - w / 2  # left coordinate
    y = cy - h / 2  # top coordinate

    bbox = {'left': x if x < size[0] else size[0], 'top': y if y < size[1] else size[1], 'width': w, 'height': h}
    return bbox

def labels_to_labelbox_labels(label_dir, image_dir, include_frames, global_keys_mapper, overwrite=False):
    annotations = []
    for frame_number in include_frames:
        frame_number = str(int(frame_number) + 1)
        file_name = 'frame' + frame_number
        frame_label_path = label_dir + '/' + file_name + '.txt'
        if not os.path.exists(frame_label_path):
            continue
        img_filename = os.path.join(image_dir, file_name + '.jpg')
        img_file = cv2.imread(img_filename, cv2.IMREAD_GRAYSCALE)
        bbox_annotations = []
        with open(frame_label_path, 'r') as frame_label_file:
            for line in frame_label_file:
                yolo_annotation = list(map(float, line.split(" ")))
                classification = 'shieldedCastle' if yolo_annotation[0] == 0 else 'unshieldedCastle'
                bbox_dm = reverse_coord_convert((img_file.shape[1], img_file.shape[0]), yolo_annotation[1:])
                bbox_annotations.append(
                    lb_types.ObjectAnnotation(
                        name=classification,
                        value=lb_types.Rectangle(
                            start=lb_types.Point(x=bbox_dm["left"], y=bbox_dm["top"]),  # x = left, y = top
                            end=lb_types.Point(x=bbox_dm["left"] + bbox_dm["width"],
                                               y=bbox_dm["top"] + bbox_dm["height"])
                            # x= left + width , y = top + height
                        )
                    )
                )
        annotations.append(
            lb_types.Label(
                data=lb_types.ImageData(global_key=global_keys_mapper[img_filename]),
                annotations=bbox_annotations
            )
        )

    return annotations


def get_latest_dataset_name(client, dataset_name_prefix="local-files-upload-"):
    datasets = filter(lambda dataset: dataset.name.startswith(dataset_name_prefix), client.get_datasets())
    dataset_names = list(map(lambda dataset: int(dataset.name.split(dataset_name_prefix)[-1]), datasets))
    latest_dataset_name = dataset_name_prefix + str((max(dataset_names) + 1 ) if len(dataset_names) > 0 else 0)
    return latest_dataset_name


def upload_images(project_id, img_paths):
    print('uploading images')
    with open('assets/credentials.json', 'r') as credentials_file:
        credentials = json.load(credentials_file)
        API_KEY = credentials["labelbox_API_KEY"]
    client = lb.Client(api_key=API_KEY)

    dataset_name = get_latest_dataset_name(client)
    new_dataset = client.create_dataset(name=dataset_name)
    dataset_id = new_dataset.uid
    print('dataset_id: ', dataset_id)
    print('dataset_name: ', dataset_name)
    try:
        task = new_dataset.create_data_rows(img_paths)
        task.wait_till_done()
    except Exception as err:
        print(f'Error while creating labelbox dataset -  Error: {err}')

    global_keys_mapper = assign_global_keys(dataset_id)

    project = client.get_project(project_id)
    print('queing datarows in project')
    batch = project.create_batch(
        "image-demo-batch",  # each batch in a project must have a unique name
        global_keys=list(global_keys_mapper.values()),  # paginated collection of data row objects, list of data row ids or global keys
        priority=1  # priority between 1(highest) - 5(lowest)
    )

    print(f"Batch: {batch}")

    return dataset_id,global_keys_mapper

def get_latest_batch_name(client, batch_name_prefix="local-files-upload"):
    pass
    # datasets = filter(lambda dataset: dataset.name.startswith(dataset_name_prefix), client.get_datasets())
    # dataset_names = list(map(lambda dataset: int(dataset.name.split(dataset_name_prefix)[-1]), datasets))
    # latest_dataset_name = dataset_name_prefix + '-' + str((max(dataset_names) + 1 ) if len(dataset_names) > 0 else 0)
    # return latest_dataset_name


def assign_global_keys(dataset_id):
    print('assigning global keys')
    with open('assets/credentials.json', 'r') as credentials_file:
        credentials = json.load(credentials_file)
        API_KEY = credentials["labelbox_API_KEY"]
    client = lb.Client(api_key=API_KEY)
    data_rows = client.get_dataset(dataset_id).data_rows()
    global_keys_map = {}
    for data_row in data_rows:
        data_row_key = str(uuid.uuid4())
        client.assign_global_keys_to_data_rows(
            [{
                "data_row_id": data_row.uid,
                "global_key": data_row_key
            }]
        )
        global_keys_map[data_row.external_id] = data_row_key

    with open('global_keys.json', 'w') as global_keys_file:
        json.dump(global_keys_map, global_keys_file)
    return global_keys_map

def get_latest_predict(runs_directory='runs/detect'):
    # get the latest predict
    predicts = glob.glob(runs_directory + '/predict*')
    predicts = list(map(os.path.basename, predicts))
    index_to_replace = predicts.index('predict')
    predicts[index_to_replace] = 'predict0'
    predicts = list(map(int, map(lambda predict: predict.split('predict')[-1], predicts)))
    max_predict = max(predicts)
    return runs_directory + '/predict' + str(max_predict)

def upload_labels(project_id, labels_dir, images_dir, include_frames, global_keys_mapper):
    with open('assets/credentials.json', 'r') as credentials_file:
        credentials = json.load(credentials_file)
        API_KEY = credentials["labelbox_API_KEY"]

    client = lb.Client(api_key=API_KEY)
    print('converting model labels')
    labels = labels_to_labelbox_labels(labels_dir, images_dir, include_frames, global_keys_mapper)

    print('starting label upload job')
    upload_job_label_import = lb.LabelImport.create_from_objects(
        client=client,
        project_id=project_id,
        name="label_import_job-" + str(uuid.uuid4()),
        labels=labels
    )

    upload_job_label_import.wait_until_done()
    print("Errors:", upload_job_label_import.errors)
    print("Status of uploads: ", upload_job_label_import.statuses)
    print("   ")


if __name__=='__main__':
    file_path = sys.argv[1]
    file_type = sys.argv[2]


    image_dir = 'datasets/images'
    if file_type == 'video':
        video_to_image_dir(file_path, image_dir)

    if len(sys.argv) > 3:
        model_path = sys.argv[3]
    else:
        model_path = 'yolov8m.pt'

    model = YOLO(model_path)
    print('creating predictions using model ', model_path)
    model(glob.glob(image_dir + '/*'), save_txt=True)

    if len(sys.argv) > 4:
        project_id = sys.argv[4]
    else:
        with open('assets/credentials.json') as credentials_file:
            credentials = json.load(credentials_file)
            project_id = credentials['labelbox_image_project_id']
    

    image_paths = glob.glob(image_dir + '/*.jpg')
    dataset_id,global_keys_mapper = upload_images(project_id, image_paths)


    upload_labels(
        project_id,
        get_latest_predict() + '/labels',
        image_dir,
        range(0, len(image_paths)),
        global_keys_mapper
    )
