#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
print 'numberof features: ', len(features_train[0])
print 'number of trains and tests', len(features_train), len(features_test)
#features_train = features_train[:len(features_train)/10] 
#labels_train = labels_train[:len(labels_train)/10] 

from sklearn.svm import SVC
clf = SVC(kernel="rbf", C=10000)

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
pred = clf.predict(features_test)
print "Prediction time:", round(time()-t1, 3), "s"

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred, labels_test)
#accuracy = model.score(y_pred, labels_test)
print accuracy
print 'Predictions: e10=', pred[10]
print 'Predictions: e26=', pred[26]
print 'Predictions: e50=', pred[50]
print '#Cris= ', sum(pred)
#########################################################



