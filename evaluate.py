from ultralytics import YOLO
import cv2
import glob
import os
import json
import labelbox as lb
import labelbox.types as lb_types
import uuid


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

    bbox = {'left': x, 'top': y, 'width': w, 'height': h}

    return bbox

def labels_to_labelbox_labels(label_dir, image_dir, include_frames, overwrite=False):
    annotations = []
    with open('global_keys.json', 'r') as global_keys_file:
        global_keys_json = json.load(global_keys_file)
    for frame_number in include_frames:
        print('Frame number:', frame_number)
        frame_number = str(int(frame_number) + 1)
        file_name = 'frame' + frame_number
        frame_label_path = label_dir + '/' + file_name + '.txt'
        if not os.path.exists(frame_label_path):
            continue
        img_file = cv2.imread(os.path.join(image_dir, file_name + '.jpg'), cv2.IMREAD_GRAYSCALE)
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
                data=lb_types.ImageData(global_key=global_keys_json[file_name + '.jpg']),
                annotations=bbox_annotations
            )
        )

    return annotations


def assign_global_keys(dataset_id):
    with open('assets/credentials.json', 'r') as credentials_file:
        credentials = json.load(credentials_file)
        API_KEY = credentials["labelbox_API_KEY"]
    client = lb.Client(api_key=API_KEY)
    data_rows = client.get_dataset(dataset_id).data_rows()
    global_keys_map = {}
    for data_row in data_rows:
        print(data_row)
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


def upload_labels(labels_dir, images_dir, include_frames):
    with open('assets/credentials.json', 'r') as credentials_file:
        credentials = json.load(credentials_file)
        API_KEY = credentials["labelbox_API_KEY"]
        project_id = credentials["labelbox_project_id"]
        # ontology_id = credentials["labelbox_ontology_id"]
        # video_url = credentials["labelbox_video_url_1"]
        # video_global_key = credentials["labelbox_video_global_key_1"]

    client = lb.Client(api_key=API_KEY)


    # global_key = video_global_key

    # test_img_url = {
    #     "row_data": video_url,
    #     "global_key": global_key
    # }
    # dataset = client.create_dataset(
    #     name="Video prediction demo",
    #     iam_integration=None  # Removing this argument will default to the organziation's default iam integration
    # )
    # task = dataset.create_data_rows([test_img_url])
    # task.wait_till_done()
    # print("Errors: ", task.errors)
    # print("Failed data rows: ", task.failed_data_rows)

    # Project defaults to batch mode with benchmark quality settings if this argument is not provided
    # Queue mode will be deprecated once dataset mode is deprecated
    # ontology = client.get_ontology(ontology_id)

    project = client.get_project(project_id)

    ## connect ontology to your project
    # project.setup_editor(ontology)



    # Python Annotation
    labels = labels_to_labelbox_labels(labels_dir, images_dir, include_frames)

    print('starting label upload job')
    upload_job_label_import = lb.LabelImport.create_from_objects(
        client=client,
        project_id=project.uid,
        name="label_import_job-" + str(uuid.uuid4()),
        labels=labels
    )

    upload_job_label_import.wait_until_done()
    print("Errors:", upload_job_label_import.errors)
    print("Status of uploads: ", upload_job_label_import.statuses)
    print("   ")

if __name__=='__main__':
    # model = YOLO('runs/detect/train5/weights/best.pt')
    # res = model(glob.glob('all_images/*'), save_txt=True)
    # upload_labels('runs/detect/predict2/labels', 'all_images', list(range(334, 335)))
    upload_labels('runs/detect/predict2/labels', 'all_images',
        list(range(191,209)) + list(range(221,271)) + list(range(276,290)) + list(range(333, 433))
    )
    # assign_global_keys("")