#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 1 (Naive Bayes) mini-project 

    use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
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


print 'numberof features: ', len(features_train[0])
print 'number of trains and tests', len(features_train), len(features_test)
#features_train = features_train[:len(features_train)/10] 
#labels_train = labels_train[:len(labels_train)/10] 

#########################################################
### your code goes here ###
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

t0 = time()
model = gnb.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
y_pred = model.predict(features_test)
print "Prediction time:", round(time()-t1, 3), "s"

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_pred, labels_test)
#accuracy = model.score(y_pred, labels_test)
print accuracy


#########################################################


