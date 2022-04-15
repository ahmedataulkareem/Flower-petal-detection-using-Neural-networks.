import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

###apply algorithms to find accuracy: 100 -- 80 -20 / 

from sklearn.linear_model import LinearRegression    ##for linear_regression
from sklearn.linear_model import LogisticRegression    ##for logistic regression
from sklearn.neighbors import KNeighborsRegressor       ##for knn regression
from sklearn.tree import DecisionTreeRegressor          ##for DecisionTreeRegression
from sklearn.ensemble import RandomForestRegressor       ##RandomForestRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

# import dataset
df = pd.read_csv('finaldataset/updatedataset.csv')
print(df.head())


# split input and output and make in array format 
x = np.array(df.drop(["Species"], axis=1)) #in
y = np.array((df["Species"])) #actualop

# check x, y
print(x)
print(y)


# split test and train data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# check train and test data length
print("x_test :",  len(x_test))
print("y_test :",  len(y_test))
print("x_trian :", len(x_train))
print("y_trian :", len(y_train))

##apply algorithms:
linear_regression = LinearRegression()
linear_regression.fit(x_train, y_train)
acc =  linear_regression.score(x_test, y_test)
print("Linar Regression : ", acc)

##apply logistic regression:

logistic_regression = LinearRegression()
logistic_regression.fit(x_train , y_train)
acc = logistic_regression.score(x_test , y_test)
print("logistic regression acc  :" , acc )

##apply k-nearest neighbours:
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(x_train , y_train)
acc = knn.score(x_test , y_test)
print("knn acc : " , acc)


##apply DecisionTreeRegressor
dtregression = DecisionTreeRegressor()
dtregression.fit(x_train , y_train)
acc = dtregression.score(x_test , y_test)
print("decision tree regression acc :" , acc)

##apply Random forest regression:
rf_regression = RandomForestRegressor()
rf_regression.fit(x_train , y_train)
acc = rf_regression.score(x_test, y_test)
print("random forest regression acc :" , acc)


##apply support vector machine regression:
svm_regression = SVR()
svm_regression.fit(x_train , y_train)
acc = svm_regression.score(x_test , y_test)
print("svm_regression acc :" , acc)


##svm by linear regressipn :
svmlinear_regression = LinearSVR()
svmlinear_regression.fit(x_train , y_train)
acc = svmlinear_regression.score(x_test , y_test)
print("svmlinear_regression acc :" , acc)


###    classifiers:   ####
##applyy knn classifier:
print("---------------classifier---------------")
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(x_train , y_train)
acc = knn_clf.score(x_test , y_test)
print("knn_clf :" , acc)

#  5
#2    1
#

##apply Decision tree clssifier:
dt_clf = DecisionTreeClassifier()
dt_clf.fit(x_train , y_train)
acc  = dt_clf.score(x_test ,y_test)
print("decisin tree clasifier:" , acc)

##apply random_forest classifier:
rf_clf = RandomForestClassifier(n_estimators=10 , criterion="entropy")
rf_clf.fit(x_train , y_train)
acc = rf_clf.score(x_test ,y_test)
print("randomforest_clf :" , acc)

##apply svm_classifier :
svm_clf = SVC()
svm_clf.fit(x_train , y_train)
acc= svm_clf.score(x_test ,y_test)
print("svm_classifier acc :" , acc)

##linear svm classifier:
linear_svc = LinearSVC()
linear_svc.fit(x_train , y_train)
acc = linear_svc.score(x_test , y_test)
print("linear_svc acc :" , acc)


## naivebayes clasifier::
naivebayes_clf = GaussianNB()
naivebayes_clf.fit(x_train ,y_train)
acc = naivebayes_clf.score(x_test , y_test)
print("naivebayes classifier acc :" ,acc)

### apply GradientBoostingClassifier:
gradientboost_clf = GradientBoostingClassifier(n_estimators=80)
gradientboost_clf.fit(x_train ,y_train)
acc =gradientboost_clf.score(x_test ,y_test)
print("GradientBoostingClassifier acc :" , acc)

