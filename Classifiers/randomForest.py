from featuresExtraction import *
from sklearn.model_selection import train_test_split
import pickle
import random
from sklearn.ensemble import RandomForestClassifier

def randomForestTrain(data_set, Y):
    X_train, X_testValid, Y_train, Y_testValid = train_test_split(data_set, Y, test_size=0.2, random_state=60)            
    features_train = getFeaturesList(X_train)
    features_test = getFeaturesList(X_testValid)

    X = np.array(features_train)
    y = np.array(Y_train)

    clf = RandomForestClassifier(random_state=12)
    clf.fit(X, y)

    pickle.dump(clf, open('./Classifiers/RF_model.pkl', 'wb'))
    model = pickle.load(open('./Classifiers/RF_model.pkl', 'rb'))

    y_pred = (clf.predict(features_train))
    acc = np.mean(y_pred == Y_train) * 100
    print("Train Data Accuracy: ",acc,'%\n')

    y_pred = (model.predict(features_test))
    acc = np.mean(y_pred == Y_testValid) * 100
    print("Test Data Accuracy: ",acc,'%\n')

    print("_________________________Training is completed_________________________")




def randomForestPredict(data_set):
    features_test = getFeaturesList(data_set)
    model = pickle.load(open('./Classifiers/RF_model.pkl', 'rb'))
    if(len(features_test)):
        y_pred = model.predict(features_test)
    else:
        y_pred= str(random.randint(0, 3))
    return y_pred