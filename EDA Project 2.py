#target variable is species=y all other columns are features=X
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("C:/Users/USER/Dev/cfehome/Iris.csv")
print(df.head())
#as we see that our y is catagorical we need to convert it to numeric
cols=df.columns
infor=df.info()
top=df.head()
end=df.tail()
#checking for errors
#REMOVING ID column as its a primary key and is unique not usefull in data
df=df.drop("Id", axis=1)
#checking for NA values
na=df.isna().sum()
print("na values are: {}".format(na))
#checking for null values
null=df.isnull().sum()
print("null values are: {}".format(null))
#checking for duplicate values
dup=df.duplicated().sum()
print("duplicated values are: {}".format(dup))
#dropping duplicates
df = df.drop_duplicates(subset=df.columns.tolist(), keep='first')
#checking for outliers
df.boxplot()
plt.title("boxplot of unclean data")
plt.show() #we see that sepal width has outliers
#checking for inconsistant dtypes
#DataType=df.dtypes()
#print(DataType)
Unic=df["Species"].unique()
print("Unique values are: {}".format(Unic))
#checking for class imbalancing
sns.countplot(x=df["Species"], data=df)
plt.show()
#cleaning outliers
quantile75=df["SepalWidthCm"].quantile(0.75)
quantile25=df["SepalWidthCm"].quantile(0.25)
iqr=quantile75-quantile25
upper=quantile75+(iqr*1.5)
lower=quantile25-(iqr*1.5)
df=df[(df["SepalWidthCm"]>lower)&(df["SepalWidthCm"]<upper)]
#boxplot after cleaning data
df.boxplot()
plt.title("box plot after cleaning")
plt.show()
#Making count plot of each column and pair plot
sns.countplot(x=df["SepalWidthCm"], data=df)
plt.title("count plot of SepalWidthCm")
plt.show()
sns.countplot(x=df["PetalLengthCm"], data=df)
plt.title("count plot of PetalLengthCm")
plt.show()
sns.countplot(x=df["PetalWidthCm"], data=df)
plt.title(" count plot of PetalWidthCm")
plt.show()
sns.pairplot(data=df, hue="Species")
plt.title("pair plot of data")
plt.show() #we see that it is showing iris setosa alag se that means iski vakue amesha kum hogi as compare to other two
#Now chaniging the categorical values of the data set Species to numeric
#Now using label encoder to make the object/string data to the boolean or integer or numeric
from sklearn.preprocessing import LabelEncoder
#MAKING isinstance
labelencode=LabelEncoder()
df["Species"]=labelencode.fit_transform(df["Species"])
print(df["Species"])
#splitting variables x, y coordinates
features=["SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
X, y=df.loc[:, features], df.loc[:, "Species"]
print(X.shape)
#now making correlation heat map
sns.heatmap(df.corr(), annot=True)
plt.title("corr heat map of the data")
plt.show()

#now splitting train stest models, using train test split model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split


XTrain, XTest, yTrain, yTest=train_test_split(X, y, test_size=0.25)
#logistic regression model

model=LogisticRegression()
#for traininng gib=ving x train and y train to the model, train k waqt diya question and answer, Y and x
model.fit(XTrain, yTrain)
#prediction or test model
#testing the learning model now, at the time of test only giving question that it will give answer and tell y pred=Test answer, we check the test answer with the actual answer, that how many anwer it gave right and wrong
yPred=model.predict(XTest)
#ytest is the actual answer and y pred is the model answer
score=accuracy_score(yTest, yPred)
accuracy=score*100
print(accuracy)

#Q)if we have a multiclass data target variable 0, 1, 2 then how can binary classification give and run model accuracy?
#Ans)If you have a multiclass classification problem with target variable values 0, 1, and 2, you should use a multiclass classification model, not binary classification. Binary classification models are designed for problems with two classes (usually 0 and 1), whereas multiclass classification models can handle more than two classes.
# But To evaluate the accuracy of a multiclass classification model, you can still use the accuracy_score function from scikit-learn. It will work for multiclass problems as well. Here's how you can calculate the accuracy score for a multiclass classification model