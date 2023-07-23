import os
import cv2


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