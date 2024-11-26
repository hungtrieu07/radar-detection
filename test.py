import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

# Functions from main.py
def find_significant_low_points(histogram, min_threshold=0.2, dip_width=10):
    histogram = histogram / histogram.max()
    peak_idx = np.argmax(histogram)

    start_idx = peak_idx
    for i in range(peak_idx, 0, -1):
        if histogram[i] < min_threshold:
            if all(histogram[max(0, i - dip_width):i] < min_threshold):
                start_idx = i
                break

    end_idx = peak_idx
    for i in range(peak_idx, len(histogram) - 1):
        if histogram[i] < min_threshold:
            if all(histogram[i:min(len(histogram), i + dip_width)] < min_threshold):
                end_idx = i
                break

    return start_idx, end_idx

def horizontal_histogram_above_threshold(image, threshold=200, normalize=False, smooth=True, median_filter_size=5, gaussian_sigma=5):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    mask = gray_image < threshold
    horizontal_histogram = np.sum(mask * gray_image, axis=1)

    if median_filter_size > 1:
        horizontal_histogram = median_filter(horizontal_histogram, size=median_filter_size)
    if smooth:
        horizontal_histogram = gaussian_filter1d(horizontal_histogram, sigma=gaussian_sigma)
    if normalize:
        horizontal_histogram = horizontal_histogram / horizontal_histogram.max()
    
    return horizontal_histogram

def vertical_histogram_above_threshold(image, threshold, normalize=False, smooth=True, median_filter_size=30, gaussian_sigma=20):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    mask = gray_image < threshold
    vertical_histogram = np.sum(mask * gray_image, axis=0)

    if median_filter_size > 1:
        vertical_histogram = median_filter(vertical_histogram, size=median_filter_size)
    if smooth:
        vertical_histogram = gaussian_filter1d(vertical_histogram, sigma=gaussian_sigma)
    if normalize:
        vertical_histogram = vertical_histogram / vertical_histogram.max()

    return vertical_histogram

# Main image processing script
image_path = r'output\\200612AA_TSGx0001_tx8-rx8_page_2_original.jpg'  # Replace with your image path
output_cropped_image = 'cropped_image.jpg'

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error loading image.")
else:
    # Calculate histograms
    horizontal_histogram = horizontal_histogram_above_threshold(image, threshold=240, normalize=True, smooth=True)
    vertical_histogram = vertical_histogram_above_threshold(image, threshold=240, normalize=True, smooth=True)

    # Find significant low points
    y_start, y_end = find_significant_low_points(horizontal_histogram, min_threshold=0.2, dip_width=20)
    x_start, x_end = find_significant_low_points(vertical_histogram, min_threshold=0.2, dip_width=20)

    # Crop the image
    cropped_image = image[y_start:y_end, x_start:x_end]

    if cropped_image.size == 0:
        print("No valid region detected for cropping.")
    else:
        # Save cropped image
        cv2.imwrite(output_cropped_image, cropped_image)
        print(f"Cropped image saved as '{output_cropped_image}'.")
