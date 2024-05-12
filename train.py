from utils import *
from preProcessing import *
from featuresExtraction import *
from Classifiers.SVM import *
from Classifiers.randomForest import *
from Classifiers.KNN import *
from Classifiers.decisionTree import *
from Classifiers.adaBoost import *

def train(inputFolder,outputFolder):
    datasetPreprocess(inputFolder,outputFolder)

    data_set , Y= readPreprocessedDataSet(outputFolder) 

    print("____________________SVM Results____________________")
    svmTrain(data_set,Y)
    print("\n")

    print("____________________RF Results____________________")
    randomForestTrain(data_set,Y)
    print("\n")

    print("____________________KNN Results____________________")
    knnTrain(data_set,Y)
    print("\n")

    print("____________________DT Results____________________")
    decisionTreeTrain(data_set,Y)
    print("\n")

    print("____________________ADABOOST Results____________________")
    adaBoostTrain(data_set,Y)
    print("\n")



train("./fonts-dataset","./preprocessed-dataset")