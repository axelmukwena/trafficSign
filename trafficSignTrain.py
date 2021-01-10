import csv

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
import pickle


def readTrafficSigns(rootpath):
    """Reads traffic sign data
    Arguments: path to the traffic sign data, for example './TrafficSignData/Training'
    Returns:   list of images, list of corresponding labels"""
    images = []  # images
    labels = []  # corresponding labels
    # loop over N classes, at most we have 42 classes
    N = 15
    for c in range(0, N):
        prefix = rootpath + '/' + format(c, '05d') + '/'  # subdirectory for class
        gtFile = open(prefix + 'GT-' + format(c, '05d') + '.csv')  # annotations file
        gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
        # gtReader.next() # skip header
        next(gtReader)
        # loop over all images in current annotations file
        for row in gtReader:
            img = Image.open(prefix + row[0])  # the 1th column is the filename
            # preprocesing image, make sure the images are in the same size
            img = img.resize((32, 32), Image.BICUBIC)
            img = np.array(img)
            images.append(img)
            labels.append(row[7])  # the 8th column is the label
        gtFile.close()
    return images, labels


# load the images
trainImages, trainLabels = readTrafficSigns('TrafficSignData/Training')
# print number of historical images
print('number of historical data=', len(trainLabels))
# show one sample image
plt.imshow(trainImages[100])
plt.show()

# design the input and output for model
X = []
Y = []
for i in range(0, len(trainLabels)):
    # input X just the flatten image, you can design other features to represent a image
    X.append(trainImages[i].flatten())
    Y.append(int(trainLabels[i]))
X = np.array(X)
Y = np.array(Y)

# train-test split specifications
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# cross-validation split using KFolds
kfold = KFold(n_splits=5)

# train SVM model & predict accuracy scores and also fit the estimator
lin_clf = svm.LinearSVC()
SVMtrainScore = cross_val_score(lin_clf, X_train, Y_train, cv=kfold)
lin_clf.fit(X_train, Y_train)
# save the SVMmodel in working directory
SVMmodel = open('SVMmodel.pkl', 'wb')
pickle.dump(lin_clf, SVMmodel)
SVMmodel.close()

# train Naive Bayes model & predict accuracy scores and also fit the estimator
gnb = GaussianNB()
GNBtrainScore = cross_val_score(gnb, X_train, Y_train, cv=kfold)
gnb.fit(X_train, Y_train)
# save the GNBmodel in working directory
GNBmodel = open('GNBmodel.pkl', 'wb')
pickle.dump(gnb, GNBmodel)
GNBmodel.close()

# train Decision Tree (DT) model & predict accuracy scores and also fit the estimator
clf = tree.DecisionTreeClassifier()
CLFtrainScore = cross_val_score(clf, X_train, Y_train, cv=kfold)
clf.fit(X_train, Y_train)
# save the CLFmodel in working directory
CLFmodel = open('CLFmodel.pkl', 'wb')
pickle.dump(clf, CLFmodel)
CLFmodel.close()

# Print the accuracy of trained data
print("Accuracy score with cross validation of specified models\n")
print("SVM Training Accuracy Score:", SVMtrainScore.mean())
print("GNB Training Accuracy Score:", GNBtrainScore.mean())
print("DT Training Accuracy Score:", CLFtrainScore.mean(), "\n")

# Test trained models

# load SVMmodel from working directory
SVMmodel = pickle.load(open('SVMmodel.pkl', 'rb'))
SVMtestScore = SVMmodel.score(X_test, Y_test)

# load GNBmodel from working directory
GNBmodel = pickle.load(open('GNBmodel.pkl', 'rb'))
GNBtestScore = GNBmodel.score(X_test, Y_test)

# load Decision Tree model from working directory
CLFmodel = pickle.load(open('CLFmodel.pkl', 'rb'))
CLFtestScore = CLFmodel.score(X_test, Y_test)

# Print the estimate of accuracy test results
print("Test accuracy score for Trained models\n")
print("SVM Test Accuracy Score:", SVMtestScore.mean())
print("GNB Test Accuracy Score:", GNBtestScore.mean())
print("DT Test Accuracy Score:", CLFtestScore.mean())
