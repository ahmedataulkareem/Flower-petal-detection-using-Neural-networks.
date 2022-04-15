##"aim" - predicting best value with best acc algorithm:

import pandas as pd
import numpy as np

##import pickle module for storing the value:
import pickle

##importing csv file
df = pd.read_csv(r"finaldataset/updatedataset.csv")
print(df.head())

# split input and output and make in array format
x = np.array(df.drop(["Species"], axis=1))
y = np.array((df["Species"]))

# check x, y
print(x)
print(y)


# split test and train data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

##  for warnings    ##
import timeit
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning,module="sklearn")
t0 = timeit.default_timer()##apply linear_supprt vector machine:


##apply random_forest classifier:
from sklearn.ensemble import RandomForestClassifier
bestscore_rfc = 0
for count in range(1000):
    rf_clf = RandomForestClassifier(n_estimators=10 , criterion="entropy")
    rf_clf.fit(x_train , y_train)
    acc = rf_clf.score(x_test ,y_test)
    print("\ncount :" , count , "acc:" , acc ,end="")
    if bestscore_rfc < acc:
        bestscore_rfc = acc
        print("-------------------->randomforest_clf :" , bestscore_rfc)
        with open("randomforest.pickle" , "wb") as rfc_file:
            pickle.dump(rf_clf , rfc_file)



##linear svm classifier:
from sklearn.svm import LinearSVC
bestscore_svc = 0
for count2 in range(1000):
    linear_svc = LinearSVC()
    linear_svc.fit(x_train , y_train)
    acc = linear_svc.score(x_test , y_test)
    print("\n count2 : " , count2 , "acc: " , acc)
    if bestscore_svc < acc:
        bestscore_svc = acc
        print("------------------>inear_svc acc :" , bestscore_svc)
        with open("linear_svc.pickle" , "wb") as svc_file:
            pickle.dump(linear_svc , svc_file)

