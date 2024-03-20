from PIL import Image
import os
import numpy as np
from skimage.metrics import mean_squared_error
from collections import defaultdict

def calculate_mse(image1, image2):

    array1 = np.array(image1)
    array2 = np.array(image2)

    if array1.shape != array2.shape:
        return float('inf') 

    mse = mean_squared_error(array1, array2)
    return mse

'''def is_subset(image1, image2):
    # Convert images to NumPy arrays
    array1 = np.array(image1)
    array2 = np.array(image2)

    # Ensure both images have the same shape
    if array1.shape != array2.shape:
        return False  # Images are not comparable

    # Create a binary mask for each image where True indicates a non-transparent pixel
    mask1 = array1[..., 3] > 0  # Assuming the images have an alpha channel at index 3
    mask2 = array2[..., 3] > 0

    # Check if all non-transparent pixels of image1 are also non-transparent in image2
    if not np.all(mask1 == (mask1 & mask2)):
        return False

    # Check if all non-transparent pixels of image1 have the same color in image2
    if not np.all(array1[mask1] == array2[mask1]):
        return False

    return True'''


def remove_duplicate_frames(input_directory, output_directory, similarity_threshold=100):
    os.makedirs(output_directory, exist_ok=True)
    similar_frames_dict = defaultdict(list)

    for filename in os.listdir(input_directory):
        if filename.endswith('.jpg'):
            frame_path = os.path.join(input_directory, filename)

            frame_image = Image.open(frame_path)
            is_duplicate = False
            for key, reference_frame in similar_frames_dict.items():
                mse = calculate_mse(frame_image, reference_frame)
                if mse < similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                similar_frames_dict[frame_path] = frame_image.copy()

    for frame_path, frame_image in similar_frames_dict.items():
        output_path = os.path.join(output_directory, os.path.basename(frame_path))
        frame_image.save(output_path)

    print(f"Total frames after removing duplicates: {len(similar_frames_dict)}")

input_frames_directory = r'C:\Users\HARSHITA KAMANI\Desktop\yt\frames'
output_frames_directory = r'C:\Users\HARSHITA KAMANI\Desktop\yt\output_frames'

remove_duplicate_frames(input_frames_directory, output_frames_directory)
