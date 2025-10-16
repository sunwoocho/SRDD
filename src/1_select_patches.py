import os
from PIL import Image
import numpy as np
from scipy.ndimage import sobel
from tqdm import tqdm
import cv2

def calculate_gradient_magnitude(image_array):
    """
    Calculates the gradient magnitude of an image using Sobel filters.

    Parameters:
        image_array (numpy.ndarray): Grayscale image as a NumPy array.

    Returns:
        float: The average gradient magnitude of the image.
    """
        # Convert to float32 if not already
    image_array = image_array.astype(np.float32)

    # Normalize if max value > 1 (assume 0~255)
    if image_array.max() > 1.0:
        image_array /= 255.0
    dx = sobel(image_array, axis=1)  # Gradient in x-direction
    dy = sobel(image_array, axis=0)  # Gradient in y-direction
    gradient_magnitude = np.hypot(dx, dy)  # Magnitude of the gradient
    return np.mean(gradient_magnitude)



def calculate_psnr(original_image, scale_factor):
    """
    Calculate PSNR between the original image and its bicubic downsampled and upsampled version.

    Parameters:
        original_image (numpy.ndarray): The original high-resolution image.
        scale_factor (float): The downscaling factor (e.g., 2 for half the size).

    Returns:
        float: The PSNR value.
    """
    # Ensure the image is in floating point for calculations
    original_image = original_image.astype(np.float32)

    # Get the dimensions of the original image
    h, w = original_image.shape[:2]

    # Compute the new dimensions for downsampling
    downsampled_h, downsampled_w = int(h / scale_factor), int(w / scale_factor)

    # Downsample the image using bicubic interpolation
    downsampled = cv2.resize(original_image, (downsampled_w, downsampled_h), interpolation=cv2.INTER_CUBIC)

    # Upsample back to original dimensions using bicubic interpolation
    upsampled = cv2.resize(downsampled, (w, h), interpolation=cv2.INTER_CUBIC)

    # Calculate Mean Squared Error (MSE)
    mse = np.mean((original_image - upsampled) ** 2)
    if mse == 0:
        return float('inf')  # Perfect match

    # Calculate PSNR
    max_pixel_value = 255.0
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))

    return psnr

def extract_patches_with_gradient_threshold(input_dir, output_dir, patch_size, step_size, psnr_threshold):
    """
    Extracts patches from images in the input directory using a sliding window approach,
    calculates their bicubic PSNR value, and saves patches exceeding the threshold as RGB images.

    Parameters:
        input_dir (str): Directory containing input images.
        output_dir (str): Directory to save the patches.
        patch_size (tuple): Size of the patches (width, height).
        step_size (int): Step size for the sliding window.
        psnr_threshold (float): Maximum PSNR value to save a patch.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    patch_width, patch_height = patch_size

    for filename in tqdm(os.listdir(input_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            image_path = os.path.join(input_dir, filename)
            image = Image.open(image_path).convert("RGB")  # Ensure images are RGB
            img_width, img_height = image.size
            
            # Skip images smaller than the patch size
            if img_width < patch_width or img_height < patch_height:
                continue

            patch_index = 0
            saved_patches = 0
            for y in range(0, img_height - patch_height + 1, step_size):
                for x in range(0, img_width - patch_width + 1, step_size):
                    patch = image.crop((x, y, x + patch_width, y + patch_height))
                    
                    # Convert the patch to grayscale for PSNR calculation
                    patch_gray = patch.convert("L")
                    patch_array = np.array(patch_gray)

                    # Calculate PSNR value magnitude
                    psnr_value = calculate_psnr(patch_array, 4)
                    if psnr_value < psnr_threshold:
                        # Save the patch as an RGB image
                        patch_filename = f"{os.path.splitext(filename)[0]}_patch_{patch_index}_{psnr_value}.png"
                        patch.save(os.path.join(output_dir, patch_filename))
                        saved_patches += 1

                    patch_index += 1


if __name__ == "__main__":
    input_home = "/path/to/input_directory"
    output_home = "/path/to/output_directory"
    dir_list = ["x4"]
    for directory in dir_list: 
        input_directory = os.path.join(input_home,directory)
        output_directory = os.path.join(output_home,directory)
        patch_size = (512, 512)  # Width, Height of each patch
        step_size = 256  # Step size for the sliding window
        psnr_threshold = 23  # Minimum average gradient magnitude to save a patch
        extract_patches_with_gradient_threshold(input_directory, output_directory, patch_size, step_size, psnr_threshold)



