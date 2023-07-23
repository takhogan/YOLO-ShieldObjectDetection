
import cv2
import ndjson
import requests
import os
import shutil
import random
import glob

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

from utils import *

# function to download video
def download_video(url, filename):
    print('downloading ', filename)
    r = requests.get(url, stream = True)
    with open(filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)

def download_image(url, filepath):
    response = requests.get(url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        # Open the file in binary mode and write the response content to it
        with open(filepath, 'wb') as file:
            file.write(response.content)
    else:
        print(f"Unable to download image. HTTP response code: {response.status_code}")


# function to convert coordinates
def coord_convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)






def process_file(filename, filetype):
    os.makedirs('datasets/labels', exist_ok=True)
    os.makedirs('datasets/images', exist_ok=True)

    with open(filename) as f:
        data = ndjson.load(f)

    for item in data:
        file_url = item['data_row']['row_data']
        if filetype == 'video':
            video_file = 'videos/' + item['data_row']['external_id']
            download_video(file_url, video_file)
            vidcap = cv2.VideoCapture(video_file)
            width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        elif filetype == 'image':
            image_filename = item['data_row']['id'] + '_' + os.path.basename(item['data_row']['external_id'])
            image_file = 'datasets/images/' + image_filename
            download_image(file_url, image_file)  # assuming download_image is implemented
            print(image_file)
            width = item['media_attributes']['width']
            height = item['media_attributes']['height']




        for project in item['projects'].values():
            for label in project['labels']:
                annotations = label['annotations']

                if filetype == 'video':
                    # Process frames
                    frames = annotations['frames']
                    for frame_number, frame_annotations in frames.items():
                        print('Frame number:', frame_number)
                        frame_number = str(int(frame_number) - 1)
                        vidcap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
                        success, image = vidcap.read()

                        if success:
                            image_file = 'datasets/images/' + item['data_row']['id'] + '_' + frame_number + '.jpg'
                            cv2.imwrite(image_file, image)

                            txt_file = 'datasets/labels/' + item['data_row']['id'] + '_' + frame_number + '.txt'
                            with open(txt_file, 'w') as f:
                                for obj_id, obj in frame_annotations['objects'].items():
                                    bbox = obj['bounding_box']
                                    x = bbox['left']
                                    y = bbox['top']
                                    w = bbox['width']
                                    h = bbox['height']
                                    bb = coord_convert((width, height), (x, x + w, y, y + h))
                                    classification = '0' if obj['name'] == 'shieldedCastle' else '1'
                                    f.write(classification + ' ' + ' '.join(map(str, bb)) + '\n')
                elif filetype == 'image':
                    # Process objects
                    label_filename = image_filename.rsplit('.', 1)[0] + '.txt'
                    label_file = 'datasets/labels/' + label_filename
                    with open(label_file, 'w') as f:
                        for obj in annotations['objects']:
                            obj_id = obj['feature_id']
                            bbox = obj['bounding_box']
                            x = bbox['left']
                            y = bbox['top']
                            w = bbox['width']
                            h = bbox['height']
                            bb = coord_convert((width, height), (x, x + w, y, y + h))
                            classification = '0' if obj['name'] == 'shieldedCastle' else '1'
                            f.write(classification + ' ' + ' '.join(map(str, bb)) + '\n')


def deprocess_file(images_dir, labels_dir):
    # List all files in the image folder
    image_files = sorted(file for file in os.listdir(images_dir) if not file.startswith('.DS_Store'))
    label_files = sorted(file for file in os.listdir(labels_dir) if not file.startswith('.DS_Store'))
    os.makedirs('figs', exist_ok=True)

    # Create a dictionary to match image and label files by their base names
    image_dict = {os.path.splitext(file)[0]: file for file in image_files}
    label_dict = {os.path.splitext(file)[0]: file for file in label_files}

    # Iterate over the common base names
    for common_name in set(image_dict.keys()).intersection(label_dict.keys()):
        # Get the image and label file paths
        img_file = image_dict[common_name]
        lbl_file = label_dict[common_name]

        # Read the image file
        print(img_file, lbl_file)
        img = cv2.imread(os.path.join(images_dir, img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib

        # Read the label file
        with open(os.path.join(labels_dir, lbl_file), 'r') as f:
            bboxes = f.readlines()

        # Create figure and axes
        fig, ax = plt.subplots(1)

        # Display the image
        ax.imshow(img)

        # for each bounding box
        for bbox in bboxes:
            # Each bbox is in format: class x_center y_center width height
            parts = bbox.strip().split()
            # convert to float
            parts = [float(part) for part in parts]
            # YOLO format is normalized to image size, let's scale it back
            x_center, y_center, width, height = parts[1] * img.shape[1], parts[2] * img.shape[0], parts[3] * \
                                                img.shape[1], parts[4] * img.shape[0]
            # Create a Rectangle patch
            rect = patches.Rectangle((x_center - width / 2, y_center - height / 2), width, height, linewidth=1,
                                     edgecolor='b' if int(parts[0]) == 0 else 'r', facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)

        plt.savefig('figs/' + os.path.splitext(os.path.basename(img_file))[0] + '-fig.jpg')
        plt.close()





def split_data(image_dir, label_dir, dataset_dir, val_percent=0.1):
    # Ensure the train and val directories exist
    os.makedirs(os.path.join(dataset_dir, 'datasets/images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'datasets/images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'datasets/labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'datasets/labels', 'val'), exist_ok=True)

    # Get a list of all image files
    images = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

    # Shuffle the image files
    random.shuffle(images)

    # Calculate the number of validation samples
    num_val = int(len(images) * val_percent)

    for i, image_file in enumerate(images):
        # Determine whether this image should be in train or val
        if i < num_val:
            target_dir = 'val'
        else:
            target_dir = 'train'

        # Construct the corresponding label file name
        label_file = image_file.rsplit('.', 1)[0] + '.txt'

        # Move the image and label files to the target directory
        shutil.move(os.path.join(image_dir, image_file), os.path.join(dataset_dir, 'datasets/images', target_dir, image_file))
        shutil.move(os.path.join(label_dir, label_file), os.path.join(dataset_dir, 'datasets/labels', target_dir, label_file))



if __name__=='__main__':
    # process_file('export-result.ndjson', 'video')
    if len(glob.glob('datasets/images/*.jpg')):
        print('Warning: images found in image folder, reccomended to keep folder empty')
    process_file('export-result-img.ndjson', 'image')
    print('finished processing')
    split_data('datasets/images', 'datasets/labels', '.')
    deprocess_file('datasets/images/train', 'datasets/labels/train')
    deprocess_file('datasets/images/val', 'datasets/labels/val')

