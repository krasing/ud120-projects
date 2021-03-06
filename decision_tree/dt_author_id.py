#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 3 (decision tree) mini-project

    use an DT to identify emails from the Enron corpus by their authors
    
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
from sklearn import tree

t0 = time()
clf = tree.DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
y_pred = clf.predict(features_test)
print "Prediction time:", round(time()-t1, 3), "s"

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_pred, labels_test)
print 'accuracy score: ', acc
print 'Decision Tree'

print 'numberof features: ', len(features_train[0])
print 'number of trains and tests', len(features_train), len(features_test)

#########################################################


