import cv2
import os

# this function reads the dataset
def readPreprocessedDataSet(input_folder):
    data_set = []
    Y = []
    image_names = []  # Store image names

    # Iterate through each category folder in the input folder
    for category_folder in os.listdir(input_folder):
        category_path = os.path.join(input_folder, category_folder)
    
        # Check if the item in the input folder is indeed a folder
        if os.path.isdir(category_path):
            # Iterate through each image in the category folder
            for image_name in os.listdir(category_path):
                image_path = os.path.join(category_path, image_name)
                image = cv2.imread(image_path)
                grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                data_set.append(grayImage)
                Y.append(os.path.basename(category_path))
                image_names.append(image_name)  # Store image name

    return (data_set, Y)        


# this function reads the dataset
def readPreprocessedTestSet(input_folder):
    data_set = []

    # Iterate through each file in the input folder
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        if os.path.isfile(file_path):  # Check if it's a file
            image = cv2.imread(file_path)
            grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            data_set.append(grayImage)

    return data_set


# this function reads the test dataset
def readTestSet(input_folder):
    test_set = []
    # Check if the item in the input folder is indeed a folder
    if os.path.isdir(input_folder):
        # Iterate through each image in the category folder
        for image in os.listdir(input_folder):
            image_path = os.path.join(input_folder, image)
            img = cv2.imread(image_path)
            test_set.append(img)

    return test_set


def clearDataLinesFolder():
    # Path to the dataLines folder
    data_lines_folder = "./dataLines"
    # Iterate over all files in the folder and remove them
    for filename in os.listdir(data_lines_folder):
        file_path = os.path.join(data_lines_folder, filename)
        os.remove(file_path)       
