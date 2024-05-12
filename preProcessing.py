import cv2
import os
import numpy as np
from imutils import contours as imcnts
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
import pytesseract
import re
import shutil
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def binarizeImage(image):
    try:
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_pixel_value = grayImage.mean()
        is_light_image = mean_pixel_value > 50
        if is_light_image:
            inverted_image = cv2.bitwise_not(grayImage)
        else:
            inverted_image = grayImage

        _, binary = cv2.threshold(inverted_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        negativeImage =255-binary      

        return negativeImage
    except Exception as e:
        return image

def rotateImage(image, angle):
    try:
        h, w = image.shape[:2]
        
        if angle >= 30 and angle <= 60:
            # Calculate rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            # Perform rotation
            rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
            return rotated_image
        else:
            return image
    except Exception as e:
        return image

def houghTransform(image):
    try:
        # Apply Gaussian blur to reduce noise
        blurred_image = cv2.GaussianBlur(image, (11, 11), 0)
        
        # Detect edges using Canny edge detection
        edges = canny(blurred_image)
        
        # Define angles to be tested for Hough transform
        tested_angles_rad  = np.deg2rad(np.arange(0.1, 180.0))
        
        # Perform Hough transform to detect lines
        hough_space, line_angles_rad, line_distances = hough_line(edges, theta=tested_angles_rad)
        
        # Find peaks in Hough space
        _, angles, _ = hough_line_peaks(hough_space, line_angles_rad, line_distances)
        
        # Rotate the image using the angle of the first detected line
        rotated_image = rotateImage(image, 180-angles[0] * 180 / np.pi)
        
        return rotated_image
    except Exception as e:
        return image

# Perform OCR orientation detection
def correctTextOrientation(image,count):
    try:
        # Explicitly set the resolution to 70 DPI to avoid warning
        rotation_data = pytesseract.image_to_osd(image, config="--dpi 70")
        
        # Extract the rotation angle from OSD data
        rotation = re.search('(?<=Rotate: )\d+', rotation_data).group(0)
        angle = float(rotation)
        
        # Adjust angle if needed (for right-to-left scripts like Arabic)
        if angle > 0:
            angle = 360 - angle
        
        # Rotate the image to deskew it
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated_image
    except pytesseract.TesseractError as e:
        if count==4:
            return image
        else:
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, 90, 1.0)
            rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            count+=1
            return correctTextOrientation(rotated_image,count)

def filterImage(image):
    try:
        filtered_image = cv2.medianBlur(image, 3)  # Adjust kernel size as needed
        return filtered_image
    except Exception as e:
        return image
    
def extractLines(image):
    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by their vertical position
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
    
    lines=[]
    current_line = []
    previous_y = -1
    
    # Iterate through contours
    for contour in contours:
        # Get bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter contours based on aspect ratio and area
        aspect_ratio = w / float(h)
        if 0.1 < aspect_ratio < 10 and cv2.contourArea(contour) > 100:
            # Check if current contour is on the same line
            if previous_y == -1 or abs(previous_y - y) < 20:
                current_line.append(contour)
            else:
                # Append the current line to lines
                lines.append(current_line)
                current_line = [contour]
            previous_y = y
    
    # Append the last line
    if current_line:
        lines.append(current_line)
    
    # Extract the lines from the image
    extracted_lines = []
    for line_contours in lines:
        line_contours = np.concatenate(line_contours)
        x, y, w, h = cv2.boundingRect(line_contours)
        line = image[y:y+h, x:x+w]
        extracted_lines.append(line)
    
    return extracted_lines


def detect_text_lines(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to binarize the image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on aspect ratio and area to extract text lines
    lines = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        if aspect_ratio > 5 and cv2.contourArea(contour) > 100:
            lines.append((x, y, x + w, y + h))
    
    return lines

#####################################################################################################################################

def datasetPreprocess(input_folder,output_folder):   
    if os.path.isdir(output_folder):
        # Remove all files and subfolders within the output folder
        shutil.rmtree(output_folder)
    # Recreate the output folder
    os.makedirs(output_folder)

    # Iterate through each category folder in the input folder
    for category_folder in os.listdir(input_folder):
        category_path = os.path.join(input_folder, category_folder)

        # Check if the item in the input folder is indeed a folder
        if os.path.isdir(category_path):
            # Create a corresponding category folder in the output folder
            output_category_path = os.path.join(output_folder, category_folder)
            if not os.path.exists(output_category_path):
                os.makedirs(output_category_path)

            foldername = os.path.basename(output_category_path)
            # Iterate through each image in the category folder
            for image_name in os.listdir(category_path):
                image_path = os.path.join(category_path, image_name)
                image = cv2.imread(image_path)

                filename = os.path.basename(image_path)
                print(filename)

                processed_image = binarizeImage(image)
                rotated_image=houghTransform(processed_image)
                rotated90=correctTextOrientation(rotated_image,0)
                filteredImage=filterImage(rotated90)
                imageLines=extractLines(filteredImage)

                # Save the preprocessed image into the corresponding category folder in the output folder
                for i,line in enumerate(imageLines):
                    output_image_path = os.path.join(output_category_path, f'image{filename.replace(".jpeg", "")}_line{i}.jpeg')
                    resized_line = cv2.resize(line, (1000, 50))
                    cv2.imwrite(output_image_path, resized_line)

            print("_____________Category ",foldername," is done_____________")

    print("_________________________Preprocessing is completed_________________________")


def imagePreprocess(image): 
    processed_image = binarizeImage(image)
    rotated_image = houghTransform(processed_image)
    rotated90 = correctTextOrientation(rotated_image,0)
    filteredImage = filterImage(rotated90)
    imageLines = extractLines(filteredImage)
    # Save the preprocessed image into the corresponding category folder in the output folder
    for i,line in enumerate(imageLines):
        output_image_path = os.path.join("./dataLines", f'line{i}.jpeg')
        resized_line = cv2.resize(line, (1000, 50))
        cv2.imwrite(output_image_path, resized_line) 