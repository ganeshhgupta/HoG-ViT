import cv2
import numpy as np
import os
from skimage.feature import hog
from tqdm import tqdm  # For the loading bar

def shift_image(image, shift_x, shift_y):
    h, w = image.shape
    #print(f"Original image shape: {image.shape}")  # Print original image dimensions
    src_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst_points = np.float32([[0, 0], [w, 0], 
                             [shift_x, h - shift_y], 
                             [w - shift_x, h - shift_y]])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    distorted_image = cv2.warpPerspective(image, M, (w, h))
    #print(f"Shifted image shape: {distorted_image.shape}")  # Print shifted image dimensions
    return distorted_image

def generate_events(image, shift_x, shift_y, threshold=40):
    shifted_image = shift_image(image, shift_x, shift_y)
    h, w = image.shape
    cropped_h = h - abs(shift_y)
    cropped_w = w - abs(shift_x)
    cropped_original = image[:cropped_h, :cropped_w]
    cropped_shifted = shifted_image[:cropped_h, :cropped_w]
    diff = cropped_shifted.astype(np.int16) - cropped_original.astype(np.int16)
    event_image = np.zeros_like(diff, dtype=np.uint8)
    event_image[diff > threshold] = 255
    event_image[diff < -threshold] = 128
    event_image = cv2.resize(event_image, (720, 480)) 
    #print(f"Event image shape: {event_image.shape}")  # Print event image dimensions
    return event_image

def save_hog_feature_map_to_npy(event_image, output_hog_path):
    # Compute HoG features with visualization
    hog_features, hog_image = hog(
        event_image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        feature_vector=False,
        channel_axis=None
    )

    #print(f"Original HoG feature map shape: {hog_features.shape}") 

    # Flatten the block dimensions (2, 2) and reduce spatial dimensions
    aggregated_hog_features = hog_features[:-1, :, 0, 0, :] + \
                              hog_features[:-1, :, 1, 1, :]  # Combine overlapping blocks
    #print(f"Modified HoG feature map shape: {aggregated_hog_features.shape}")

    # Save the modified HoG feature map
    np.save(output_hog_path, aggregated_hog_features)

def process_images(input_folder, output_event_folder, output_hog_folder, max_images=3626):
    if not os.path.exists(output_event_folder):
        os.makedirs(output_event_folder)
    if not os.path.exists(output_hog_folder):
        os.makedirs(output_hog_folder)

    image_files = [f for f in os.listdir(input_folder) if f.endswith((".png", ".jpg", ".jpeg"))]
    total_images = min(len(image_files), max_images)  # Limit to max_images (3 images)

    with tqdm(total=total_images, desc="Processing Images", unit="image") as pbar:
        for i, filename in enumerate(image_files[:total_images]):
            input_path = os.path.join(input_folder, filename)
            image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

            # Print original image shape
            #print(f" Original image {i+1} ({filename}) shape: {image.shape}")
            
            # Resize image to one-fourth of its original size
            height, width = image.shape
            #resized_image = cv2.resize(image, (width // 1, height // 1)) # /1
            resized_image = cv2.resize(image, (720, 480)) 

            # Print resized image shape
            #print(f" Resized image {i+1} shape: {resized_image.shape}")

            # Generate event-based image
            event_image = generate_events(resized_image, shift_x=2, shift_y=5, threshold=20)
            event_filename = os.path.splitext(filename)[0] + os.path.splitext(filename)[1]
            output_event_path = os.path.join(output_event_folder, event_filename)
            cv2.imwrite(output_event_path, event_image)
            
            # Generate and save HoG feature map
            hog_filename = os.path.splitext(filename)[0] + ".txt"
            output_hog_path = os.path.join(output_hog_folder, hog_filename)
            save_hog_feature_map_to_npy(event_image, output_hog_path)

            pbar.update(1)

def process_masks(input_folder, output_event_folder, output_hog_folder, max_images=3626):
    # if not os.path.exists(output_event_folder):
    #     os.makedirs(output_event_folder)
    if not os.path.exists(output_hog_folder):
        os.makedirs(output_hog_folder)

    image_files = [f for f in os.listdir(input_folder) if f.endswith((".png", ".jpg", ".jpeg"))]
    total_images = min(len(image_files), max_images)  # Limit to max_images (3 images)

    with tqdm(total=total_images, desc="Processing Images", unit="image") as pbar:
        for i, filename in enumerate(image_files[:total_images]):
            input_path = os.path.join(input_folder, filename)
            image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

            # Print original image shape
            print(f" Original image {i+1} ({filename}) shape: {image.shape}")
            
            # # Resize image to one-fourth of its original size
            # height, width = image.shape
            # #resized_image = cv2.resize(image, (width // 1, height // 1)) # /1
            image = cv2.resize(image, (720, 480)) 

            # # Print resized image shape
            # print(f" Resized image {i+1} shape: {resized_image.shape}")

            # # Generate event-based image
            # event_image = generate_events(resized_image, shift_x=2, shift_y=5, threshold=20)
            # event_filename = os.path.splitext(filename)[0] + os.path.splitext(filename)[1]
            # output_event_path = os.path.join(output_event_folder, event_filename)
            # cv2.imwrite(output_event_path, event_image)
            
            # Generate and save HoG feature map
            hog_filename = os.path.splitext(filename)[0] + ".txt"
            output_hog_path = os.path.join(output_hog_folder, hog_filename)
            save_hog_feature_map_to_npy(image, output_hog_path)

            pbar.update(1)

if __name__ == "__main__":
    # Folders for input images and outputs
    input_folder = "./frames"  # Place your input images in this folder
    mask_input_folder = "./lane-masks"  # Place your input images in this folder
    output_event_folder = "./event_images"
    mask_output_event_folder = "./event_masks"
    output_hog_folder = "./hog_features"
    mask_output_hog_folder = "./hog_masks"
    # Process only the first 3 images
    #process_images(input_folder, output_event_folder, output_hog_folder, max_images=3626)
    process_masks(mask_input_folder, mask_output_event_folder, mask_output_hog_folder, max_images=3626)
