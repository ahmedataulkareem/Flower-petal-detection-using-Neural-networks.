import pickle

## randomforest classification:
rf_model = pickle.load(open('randomforest.pickle', "rb"))

sepallength = float(input("Enter sepal lenght: "))
sepalwidth = float(input("Enter sepal width: "))
petallength = float(input("Enter petal length: "))
petalwidth = float(input("Enter petal width: "))
print("Pred Cases using(random forest clf) : ",rf_model.predict([[sepallength ,sepalwidth,petallength,petalwidth]]))


#### linear_svc classification:
linearsvc_model =pickle.load(open("linear_svc.pickle" , "rb"))

sepallength = float(input("Enter sepal lenght: "))
sepalwidth = float(input("Enter sepal width: "))
petallength = float(input("Enter petal length: "))
petalwidth = float(input("Enter petal width: "))

print("Pred Cases using(linear_svc) : ",linearsvc_model.predict([[sepallength ,sepalwidth ,petallength , petalwidth]]))