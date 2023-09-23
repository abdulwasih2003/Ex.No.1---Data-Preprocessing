# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train

## PROGRAM:-

Name: Sarankumar J

Reg No: 212221230087
```py
#importing libraries
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Reading the dataset
df=pd.read_csv("/content/Churn_Modelling.csv", index_col="RowNumber")
df

#Dropping the unwanted Columns
df.drop(['CustomerId'],axis=1,inplace=True)
df.drop(['Surname'],axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df

#Checking for null values
df.isnull().sum()

#Checking for duplicate values
df.duplicated()

#Describing the dataset
df.describe()

#Scaling the dataset
scaler=StandardScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
df1

#Allocating X and Y attributes
x=df1.iloc[:,:-1].values
x
y=df1.iloc[:,-1].values
y

#Splitting the data into training and testing dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```
## OUTPUT:-

The dataset

![op1](https://github.com/SarankumarJ/Ex.No.1---Data-Preprocessing/assets/94778101/c4e70658-7328-4e9f-b85b-143e8400043b)

Dropping unwanted features

![op2](https://github.com/SarankumarJ/Ex.No.1---Data-Preprocessing/assets/94778101/f83dd65e-0644-42e4-8754-fd3099d64c25)


Checking for null values

![op3](https://github.com/SarankumarJ/Ex.No.1---Data-Preprocessing/assets/94778101/2c517b1d-a8f3-4cf2-b5f7-658c2db97dcc)


Checking for duplication

![op4](https://github.com/SarankumarJ/Ex.No.1---Data-Preprocessing/assets/94778101/6f74b0e8-00a0-4408-8c3f-10be509c735d)


Describing the dataset

![op5](https://github.com/SarankumarJ/Ex.No.1---Data-Preprocessing/assets/94778101/2bdd1870-5288-4144-875f-0daca4187b26)

Scaling the values

![op6](https://github.com/SarankumarJ/Ex.No.1---Data-Preprocessing/assets/94778101/bce4b62f-6fab-4e64-9071-53c1279f1ab7)

X Features

![op7](https://github.com/SarankumarJ/Ex.No.1---Data-Preprocessing/assets/94778101/9bd0495a-2d1b-4ec9-8bc7-3ea0e4bc8091)

Y Features

![op8](https://github.com/SarankumarJ/Ex.No.1---Data-Preprocessing/assets/94778101/041d5778-eafe-4499-9bce-4279905d6a5b)


Splitting the training and testing dataset

![op9](https://github.com/SarankumarJ/Ex.No.1---Data-Preprocessing/assets/94778101/5109a96c-7345-4c24-b27b-03cb07256505)


## RESULT
Thus we have successfully performed Data preprocessing in a data set downloaded from Kaggle
