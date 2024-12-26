import fitz  # PyMuPDF
import cv2
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d, median_filter

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

def process_pdf(pdf_path, output_folder, max_intensity_threshold=0.86):
    pdf_document = fitz.open(pdf_path)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    os.makedirs(output_folder, exist_ok=True)

    total_pages = len(pdf_document)

    for page_number in range(total_pages):
        page = pdf_document[page_number]
        pix = page.get_pixmap(dpi=600, colorspace=fitz.csGRAY)
        page_image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        
        cv2.imwrite(os.path.join(output_folder, f"{pdf_name}_page_{page_number+1}_original.jpg"), page_image)

        # Calculate histograms
        horizontal_histogram = horizontal_histogram_above_threshold(page_image, threshold=240, normalize=True, smooth=True)
        vertical_histogram = vertical_histogram_above_threshold(page_image, threshold=240, normalize=True, smooth=True)

        # Find significant low points around the main peak in both histograms
        y_start, y_end = find_significant_low_points(horizontal_histogram, min_threshold=0.2, dip_width=20)
        x_start, x_end = find_significant_low_points(vertical_histogram, min_threshold=0.2, dip_width=20)

        # Extract detected region and calculate intensity
        detected_region = page_image[y_start:y_end, x_start:x_end]
        if detected_region.size == 0:
            continue

        normalized_region_intensity = np.sum(detected_region) / (detected_region.size * 255)
        if normalized_region_intensity > max_intensity_threshold:
            print(f"Skipping page {page_number + 1} due to high intensity ({normalized_region_intensity:.2f}).")
            continue  # Skip this page if intensity is too high

        # Save cropped image (processed image)
        cropped_filename = os.path.join(output_folder, f"{pdf_name}_page_{page_number+1}_cropped.jpg")
        cv2.imwrite(cropped_filename, detected_region)

    pdf_document.close()
    print(f"Processed images (cropped) are saved in '{output_folder}'.")

# Example usage with all PDF files in a folder
pdf_folder = '26122024'

output_folder = 'output'

for pdf_file in os.listdir(pdf_folder):
    if pdf_file.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        process_pdf(pdf_path, output_folder, max_intensity_threshold=0.86)

# Example usage with a single PDF file
# pdf_file_path = r'pdf\200612AA_TSGx0001_tx8-rx8.pdf'
# output_folder = 'output'
# process_pdf(pdf_file_path, output_folder, max_intensity_threshold=0.86)
