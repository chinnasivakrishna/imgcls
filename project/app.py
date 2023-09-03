import os
import cv2
import numpy as np
from flask import Flask, request, render_template, send_file

app = Flask(__name__)
UPLOAD_FOLDER = 'uploaded_images'
PROCESSED_FOLDER = 'static/processed_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

def process_image(file_path):
    # Your image processing code here
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)

    if image is None:
        return None

    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filtering to enhance edges while removing noise
    filtered_image = cv2.bilateralFilter(grayscale_image, d=9, sigmaColor=75, sigmaSpace=75)

    # Apply adaptive thresholding to create a binary image
    threshold_image = cv2.adaptiveThreshold(filtered_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 2)

    # Apply erosion to eliminate small dots
    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.erode(threshold_image, kernel, iterations=1)

    # Find the contours in the eroded image
    contours, _ = cv2.findContours(eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank white image with the same size as the original image
    sketch_image = np.ones_like(image) * 255

    # Draw each contour on the blank image with black color (0) and thicker lines (3)
    for contour in contours:
        cv2.drawContours(sketch_image, [contour], -1, (0), 3)

    # Create a mask to isolate the eye globe area
    if len(contours) > 0:
        mask = np.zeros_like(grayscale_image)
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

        # Change the eye globe area to dim black using the mask
        sketch_image[mask == 255] = 50  # Change the intensity value here

    return sketch_image

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file part"

    file = request.files['image']

    if file.filename == '':
        return "No selected file"

    if file:
        # Save the uploaded file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Process the uploaded image
        processed_image = process_image(filename)

        if processed_image is not None:
            # Save the processed image
            processed_filename = os.path.join(app.config['PROCESSED_FOLDER'], 'processed.jpg')
            cv2.imwrite(processed_filename, processed_image)

            # Send the processed image as a response
            return send_file(processed_filename, mimetype='image/jpeg')
        else:
            return "Image processing failed"

if __name__ == '__main__':
    app.run(debug=True)
