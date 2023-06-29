
import cv2
import ndjson
import requests
import os
import shutil
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
sys.path.append('./JSON2YOLO-master')

# function to download video
def download_video(url, filename):
    print('downloading ', filename)
    r = requests.get(url, stream = True)
    with open(filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)


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



def extract_frames(video_file, output_dir):
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    vidcap = cv2.VideoCapture(video_file)

    image_counter = 0

    while vidcap.isOpened():
        # Read a frame from the video
        success, image = vidcap.read()

        if success:
            # Write the image to a file
            cv2.imwrite(os.path.join(output_dir, f'frame{image_counter}.jpg'), image)
            image_counter += 1
        else:
            # If we didn't successfully read a frame, we're probably at the end of the video
            break

    # Release the video capture
    vidcap.release()

    print(f"Saved {image_counter} images from video file {video_file}")


def process_file(filename):
    os.makedirs('labels', exist_ok=True)
    os.makedirs('images', exist_ok=True)

    with open(filename) as f:
        data = ndjson.load(f)

    for item in data:
        video_url = item['data_row']['row_data']
        video_file = 'videos/' + item['data_row']['external_id']
        download_video(video_url, video_file)
        vidcap = cv2.VideoCapture(video_file)
        extract_frames(video_file, 'all_images')
        width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        for project in item['projects'].values():
            for label in project['labels']:
                annotations = label['annotations']
                frames = annotations['frames']
                # segments = annotations.get('segments', {})

                # Process frames
                for frame_number, frame_annotations in frames.items():
                    print('Frame number:', frame_number)
                    frame_number = str(int(frame_number) - 1)
                    vidcap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
                    success, image = vidcap.read()

                    if success:
                        image_file = 'images/' + item['data_row']['id'] + '_' + frame_number + '.jpg'
                        cv2.imwrite(image_file, image)

                        txt_file = 'labels/' + item['data_row']['id'] + '_' + frame_number + '.txt'
                        with open(txt_file, 'w') as f:
                            for obj_id, obj in frame_annotations['objects'].items():
                                bbox = obj['bounding_box']
                                x = bbox['left']
                                y = bbox['top']
                                w = bbox['width']
                                h = bbox['height']
                                bb = coord_convert((width, height), (x, x + w, y, y + h))
                                f.write('0 ' + ' '.join(map(str, bb)) + '\n')


def deprocess_file():
    # List all files in the image folder
    image_files = sorted(file for file in os.listdir('images') if not file.startswith('.DS_Store'))
    label_files = sorted(file for file in os.listdir('labels') if not file.startswith('.DS_Store'))
    os.makedirs('figs', exist_ok=True)
    # Iterate over the image and corresponding label files
    for img_file, lbl_file in zip(image_files, label_files):
        # Read the image file
        print(img_file, lbl_file)
        img = cv2.imread(os.path.join('images', img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib

        # Read the label file
        with open(os.path.join('labels', lbl_file), 'r') as f:
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
                                     edgecolor='r', facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)

        plt.savefig('figs/' + os.path.splitext(os.path.basename(img_file))[0] + '-fig.jpg')
        plt.close()


import os
import shutil
import random

def split_data(image_dir, label_dir, dataset_dir, val_percent=0.1):
    # Ensure the train and val directories exist
    os.makedirs(os.path.join(dataset_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'labels', 'val'), exist_ok=True)

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
        shutil.move(os.path.join(image_dir, image_file), os.path.join(dataset_dir, 'images', target_dir, image_file))
        shutil.move(os.path.join(label_dir, label_file), os.path.join(dataset_dir, 'labels', target_dir, label_file))




process_file('export-result.ndjson')
print('finished processing')
deprocess_file()

split_data('./images', './labels', '.')



