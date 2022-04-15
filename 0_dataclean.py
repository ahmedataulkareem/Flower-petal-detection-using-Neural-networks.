##import data by using pandas module:
import pandas as pd

df = pd.read_csv("rawdataset/Iris.csv")
print(df.head())

# import first & last 5 rows
print(df.head())
print(df.tail())

##find any null value (sum of column null values):
print("\n\n null values :",df.isnull())
print("\n\n sum of all null values" ,df.isnull().sum())

##mean, median values:
print(df.describe())

##display columns names :
print(df.columns)

# checking dimension (num of rows and columns) of dataset
print("\n\n iris data shape (Rows, Columns):", df.shape)


##ploting graphs:
#for ploting we use matplotlib module
import matplotlib.pyplot as plt
import seaborn as sns
x = df['SepalLengthCm']
y = df['Species']
plt.xlabel('SepalLengthCm')
plt.ylabel('Species')
plt.scatter(x , y)
plt.show()

#3ploting by SepalWidthCm
x = df['SepalWidthCm']
y = df['Species']
plt.xlabel('SepalWidthCm')
plt.ylabel('Species')
plt.scatter(x , y)
plt.show()

##ploting by PetalLengthCm
x = df['PetalLengthCm']
y = df['Species']
plt.xlabel('PetalLengthCm')
plt.ylabel('Species')
plt.scatter(x , y)
plt.show()


##ploting by PetalWidthCm
x = df['PetalWidthCm']
y = df['Species']
plt.xlabel('PetalWidthCm')
plt.ylabel('Species')
plt.scatter(x , y)
plt.show()

###identifying unique sepallength values:
print(df['SepalLengthCm'].nunique())
plt.figure(figsize=(10,7))

print(df['SepalLengthCm'].value_counts())
df['SepalLengthCm'].value_counts().plot.bar()
plt.show()


##clasification of species using Graph:
plt.figure(figsize=(20, 6))
cols = ['yellowgreen', 'lightcoral','gold']
plt.subplot(1,2,1)
sns.countplot('Species',data=df, palette='Set1')
plt.title('Iris Species Count',fontweight="bold", size=20)
plt.xticks(fontweight="bold")

plt.subplot(1,2,2)
df['Species'].value_counts().plot.pie(explode=[0.05,0.05,0.1],autopct='%1.1f%%',shadow=True, colors=cols)
plt.title('Iris Species Count',fontweight="bold", size=20)
plt.xticks(fontweight="bold")
plt.show()


# Boxplot
plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.boxplot(x='Species', y='SepalLengthCm', data=df)
plt.subplot(2,2,2)
sns.boxplot(x='Species', y='SepalWidthCm', data=df)
plt.subplot(2,2,3)
sns.boxplot(x='Species', y='PetalLengthCm', data=df)
plt.subplot(2,2,4)
sns.boxplot(x='Species', y='PetalWidthCm', data=df)
plt.show()


##for replace target values str to num by(dicS : values)
from pandas import DataFrame
species = {"Iris-setosa" : 0,
           "Iris-versicolor" :1,
           "Iris-virginica":2

}
df["Species"] = df["Species"].replace(species)

##for verifying check
print(df.head())

# export data to new .csv file
DataFrame(df , columns= ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm',
       'Species']).to_csv("finaldataset/updatedataset.csv" ,index=False ,header=True)
print("successfully updated")
