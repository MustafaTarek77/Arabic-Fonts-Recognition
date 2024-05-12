from featuresExtraction import *
from sklearn.model_selection import train_test_split
import pickle
import random
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def knnTrain(data_set, Y):
    X_train, X_testValid, Y_train, Y_testValid = train_test_split(data_set, Y, test_size=0.2, random_state=60)            
    features_train = getFeaturesList(X_train)
    features_test = getFeaturesList(X_testValid)

    X = np.array(features_train)
    y = np.array(Y_train)

    clf = KNeighborsClassifier()
    clf.fit(X, y)

    pickle.dump(clf, open('./Classifiers/KNN_model.pkl', 'wb'))
    model = pickle.load(open('./Classifiers/KNN_model.pkl', 'rb'))

    y_pred_train = clf.predict(features_train)
    acc_train = np.mean(y_pred_train == Y_train) * 100
    print("Train Data Accuracy: ", acc_train, '%\n')

    y_pred_test = model.predict(features_test)
    acc_test = np.mean(y_pred_test == Y_testValid) * 100
    print("Test Data Accuracy: ", acc_test, '%\n')

    print("_________________________Training is completed_________________________")

def knnPredict(data_set):
    features_test = getFeaturesList(data_set)
    model = pickle.load(open('./Classifiers/KNN_model.pkl', 'rb'))
    if(len(features_test)):
        y_pred = model.predict(features_test)
    else:
        y_pred= str(random.randint(0, 3))
    return y_pred