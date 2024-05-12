from numpy import argmax
import numpy as np
import glob
import time
from utils import *
from preProcessing import *
from featuresExtraction import *
from Classifiers.SVM import *
from Classifiers.randomForest import *
from Classifiers.KNN import *
from Classifiers.decisionTree import *
from Classifiers.adaBoost import *

# Fill your team number here
TEAM_NUM = "1"

def make_prediction(img):
    # DEFINE YOUR FUNCTION HERE AND DO NOT CHANGE ANYTHING ELSE IN THE NOTEBOOK

    imagePreprocess(img)
    data_set = readPreprocessedTestSet("./dataLines")
    clearDataLinesFolder()
    labels = svmPredict(data_set)
    votes = [0, 0, 0, 0]
    for label in labels:
        label = int(label)
        votes[label] += 1
    max_label = argmax(votes)

    return max_label

def predict():
    ## 1. Fill x_test and y_test:
    x_test = []
    y_test = []
    fonts = [ 'IBM Plex Sans Arabic', 'Lemonada', 'Marhey', 'Scheherazade New']

    for font in fonts:
        for filename in sorted(glob.glob(f'./data/{font}/*.jpeg')):
            img = cv2.imread(filename, cv2.COLOR_BGR2RGB)
            x_test.append(img)
            y_test.append(fonts.index(font))

    # 2. Convert them to Numpy arrays:
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    print("shape of x_data:", x_test.shape, "shape of y_data:", y_test.shape)
    assert set(y_test) == {0, 1, 2, 3}

    y_pred = []
    if os.path.isdir("./dataLines"):
        os.rmdir("./dataLines")
    os.makedirs("./dataLines")
    start_time = time.time()
    for x in x_test:
        assert x.shape == (1181, 1181, 3)
        ŷ = make_prediction(x)
        y_pred.append(ŷ)
    end_time = time.time()
    os.rmdir("./dataLines")

    y_pred = np.asarray(y_pred)
    accuracy = np.mean(y_pred == y_test)
    total_time = end_time - start_time
    print(f"Team {TEAM_NUM} got accuracy: {accuracy:.2%}")
    print(f"Team {TEAM_NUM} got runtime: {total_time:.2%}")



def predictImage(img):
    fonts = [ 'IBM Plex Sans Arabic', 'Lemonada', 'Marhey', 'Scheherazade New']
    if os.path.isdir("./dataLines"):
        os.rmdir("./dataLines")
    os.makedirs("./dataLines")
    font = make_prediction(img)
    os.rmdir("./dataLines")
    return fonts[font]


def test(input_folder):
    test_set = readTestSet(input_folder)

    os.makedirs("./dataLines")
    # Open the results and times files in append mode
    with open('Results.txt', 'w') as results_file, open('Times.txt', 'w') as times_file:
        # Iterate through each image in the input folder
        for image in test_set:
            start_time = time.time()  
            imagePreprocess(image)
            data_set = readPreprocessedTestSet("./dataLines")
            clearDataLinesFolder() 

            labels = svmPredict(data_set)

            votes = [0, 0, 0, 0]
            for label in labels:
                label = int(label)
                votes[label] += 1

            max_label = argmax(votes)
            
            results_file.write(f"{max_label}\n")

            end_time = time.time()  # End time
            elapsed_time = end_time - start_time 
            
            rounded_time = round(elapsed_time, 3)
            times_file.write(f"{rounded_time}\n")


    os.rmdir("./dataLines")
