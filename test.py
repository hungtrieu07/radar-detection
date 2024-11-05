import cv2
import numpy as np

# Load the image
image_path = r'output\extracted_regions\240511AC-LIDx0008_page_1_original.jpg'  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded correctly
if image is None:
    print("Error loading image.")
else:
    # Define a kernel (structuring element) for erosion and dilation
    kernel_size = (3, 3)  # You can adjust the kernel size as needed
    kernel = np.ones(kernel_size, np.uint8)

    # Set the number of iterations for erosion and dilation
    erosion_iterations = 2
    dilation_iterations = 7

    # Perform erosion
    # eroded_image = cv2.erode(image, kernel, iterations=erosion_iterations)

    # Perform dilation
    dilated_image = cv2.dilate(image, kernel, iterations=dilation_iterations)

    # Save eroded and dilated images
    # cv2.imwrite('eroded_image.jpg', eroded_image)
    cv2.imwrite('dilated_image.jpg', dilated_image)
    print("Eroded and dilated images have been saved.")

    # Find contours in the dilated image
    contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours were found
    if contours:
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the bounding rectangle for the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Draw the rectangle on a copy of the dilated image
        dilated_image_with_rect = cv2.cvtColor(dilated_image, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(dilated_image_with_rect, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red rectangle with thickness 2

        # Save the image with the rectangle
        cv2.imwrite('dilated_with_largest_contour_rect.jpg', dilated_image_with_rect)
        print("Largest contour rectangle image has been saved as 'dilated_with_largest_contour_rect.jpg'.")

        # Optionally, print the coordinates and dimensions of the rectangle
        print(f"Largest contour rectangle coordinates: (x={x}, y={y}, w={w}, h={h})")
    else:
        print("No contours found in the dilated image.")
