import csv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
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


# load the images for testing
testImages, testLabels = readTrafficSigns('TrafficSignData/Training')
# print number of historical images
print('number of historical data=', len(testLabels))
# show one sample image
plt.imshow(testImages[500])
plt.show()

# design the input and output for model
X = []
Y = []
for i in range(0, len(testLabels)):
    # input X just the flatten image, you can design other features to represent a image
    X.append(testImages[i].flatten())
    Y.append(int(testLabels[i]))
X = np.array(X)
Y = np.array(Y)

# train-test split specifications
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1)

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

# test the accuracy of trained data
print("\nAccuracy score test on trained data of specified models")
print("SVM Accuracy Score:", SVMtestScore.mean())
print("GNB Accuracy Score:", GNBtestScore.mean())
print("DT Accuracy Score:", CLFtestScore.mean())
