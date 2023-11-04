#-------------------------------------------------------------------------
# AUTHOR: Mark Haddad
# FILENAME: svm.py
# SPECIFICATION: Creating several combinations of SVMs, find the best accuracy among them
# FOR: CS 4210- Assignment #3
# TIME SPENT: 45 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import svm
import numpy as np
import pandas as pd

#defining the hyperparameter values
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the training data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature training data and convert them to NumPy array
y_training = np.array(df.values)[:,-1] #getting the last field to create the class training data and convert them to NumPy array

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the training data by using Pandas library

X_test = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature testing data and convert them to NumPy array
y_test = np.array(df.values)[:,-1] #getting the last field to create the class testing data and convert them to NumPy array

#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
#--> add your Python code here
highest_accuracy = 0

for val in c:
    for deg in degree:
        for kern in kernel:
           for func_shape in decision_function_shape:

                #Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape.
                #For instance svm.SVC(c=1, degree=1, kernel="linear", decision_function_shape = "ovo")
                #--> add your Python code here
                clf = svm.SVC(C=val, degree=deg, kernel=kern, decision_function_shape=func_shape)

                #Fit SVM to the training data
                #--> add your Python code here
                clf.fit(X_training, y_training)

                #make the SVM prediction for each test sample and start computing its accuracy
                #hint: to iterate over two collections simultaneously, use zip()
                #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
                #to make a prediction do: clf.predict([x_testSample])
                #--> add your Python code here
                correct = 0
                total = 0
                predictions = clf.predict(X_test)
                for (x_testSample, y_testSample, prediction) in zip(X_test, y_test, predictions):
                    if prediction == y_testSample:
                        correct += 1
                        # print(f"Predicton: {prediction}")
                    total += 1

                #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
                #with the SVM hyperparameters. Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                #--> add your Python code here
                accuracy = correct/total
                if accuracy >= highest_accuracy:
                    highest_accuracy = accuracy
                print(f"Highest SVM accuracy so far: {highest_accuracy}, Parameters: c={val}, degree={deg}, kernel={kern}, decision_function_shape={func_shape}")

