# EX-05-Feature-Generation


## AIM
To read the given data and perform Feature Generation process and save the data to a file. 

# Explanation
Feature Generation (also known as feature construction, feature extraction or feature engineering) is the process of transforming features into new features that better relate to the target.
 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Generation techniques to all the feature of the data set
### STEP 4
Save the data to the file


# CODE
## Data.csv :
```
import pandas as pd
df=pd.read_csv("data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
import category_encoders as ce
be=ce.BinaryEncoder()
ohe=OneHotEncoder(sparse=False)
le=LabelEncoder()
oe=OrdinalEncoder()


df1["City"] = ohe.fit_transform(df1[["City"]])

temp=['Cold','Warm','Hot','Very Hot']
oe1=OrdinalEncoder(categories=[temp])
df1['Ord_1'] = oe1.fit_transform(df1[["Ord_1"]])

edu=['High School','Diploma','Bachelors','Masters','PhD']
oe2=OrdinalEncoder(categories=[edu])
df1['Ord_2']= oe2.fit_transform(df1[["Ord_2"]])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df5
```
## Encoding.csv :
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
le=LabelEncoder()
oe=OrdinalEncoder()

df1["nom_0"] = oe.fit_transform(df1[["nom_0"]])
temp=['Cold','Warm','Hot']
oe2=OrdinalEncoder(categories=[temp])
df1['ord_2'] = oe2.fit_transform(df1[['ord_2']])

df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df0=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df0

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df2=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df2

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df3=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df3

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df4=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df4
```
## Titanic.csv :
```
import pandas as pd
df=pd.read_csv("titanic_dataset.csv")
df

#removing unwanted data
df.drop("Name",axis=1,inplace=True)
df.drop("Ticket",axis=1,inplace=True)
df.drop("Cabin",axis=1,inplace=True)

#data cleaning
df.isnull().sum()

df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])

df.isnull().sum()

df

#feature encoding
from category_encoders import BinaryEncoder
be=BinaryEncoder()
df["Sex"]=be.fit_transform(df[["Sex"]])
ndf=be.fit_transform(df["Sex"])
ndf

df1=df.copy()
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
embark=['S','C','Q']
e1=OrdinalEncoder(categories=[embark])
df1['Embarked'] = e1.fit_transform(df[['Embarked']])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df5
```

# OUPUT
## Data.csv :

![image](https://user-images.githubusercontent.com/66360846/195596531-b68f0526-f0ec-4259-9a67-0f0bb8b02847.png)

![image](https://user-images.githubusercontent.com/66360846/195596571-6a58411d-73bd-456e-b706-81cac98f48d9.png)

![image](https://user-images.githubusercontent.com/66360846/195596597-ba3d9fd3-796a-4c5c-932f-6257a18fe190.png)

![image](https://user-images.githubusercontent.com/66360846/195596696-b2223a0e-2e04-49ce-a4c0-331f8f7aee33.png)

![image](https://user-images.githubusercontent.com/66360846/195596731-d044ee9d-0951-47f4-96aa-a220fd7bee52.png)

![image](https://user-images.githubusercontent.com/66360846/195596763-70246f26-3b05-471b-9be9-47e7f2c2c61b.png)

![image](https://user-images.githubusercontent.com/66360846/195596792-20bb802b-255a-4a6f-9009-9c14219fd411.png)

![image](https://user-images.githubusercontent.com/66360846/195596834-9cb3d404-c62b-4cbc-a01f-f8316298c73b.png)

## Encoding.csv :

![image](https://user-images.githubusercontent.com/66360846/195596934-3052bedc-0664-4b01-943a-77a2274efe41.png)

![image](https://user-images.githubusercontent.com/66360846/195596913-8e17a687-de64-4911-b13c-126f1c20e40b.png)

![image](https://user-images.githubusercontent.com/66360846/195596998-f1c64537-f01b-46d6-bd16-f66b3e720e19.png)

![image](https://user-images.githubusercontent.com/66360846/195597022-9435ebd0-a3f6-4758-9065-dc3494d91397.png)

![image](https://user-images.githubusercontent.com/66360846/195597091-c99b4dd7-ed51-4489-a797-a3a6a13cb3d3.png)

![image](https://user-images.githubusercontent.com/66360846/195597137-44a47720-6356-41b4-a4d5-8c2a924da4a6.png)

![image](https://user-images.githubusercontent.com/66360846/195597215-f3a627f7-a110-441c-8370-4478da3db87c.png)

![image](https://user-images.githubusercontent.com/66360846/195597324-6a23a40b-ff9a-45e6-aeea-628d9e8f0a47.png)

## Titanic.csv :
![image](https://user-images.githubusercontent.com/66360846/195597358-07e6b926-39d1-44b2-81d6-087777fbc203.png)

![image](https://user-images.githubusercontent.com/66360846/195597403-862ae4a8-5805-4a32-835f-0fa608f1ab9c.png)

![image](https://user-images.githubusercontent.com/66360846/195597435-7c927495-ce46-46b3-855f-6efac1a1cb4d.png)

![image](https://user-images.githubusercontent.com/66360846/195597455-00c4c28e-aa71-4e34-8502-749129782286.png)

![image](https://user-images.githubusercontent.com/66360846/195597487-15f3498b-3c98-47e8-9de4-55b2e2ef8880.png)

![image](https://user-images.githubusercontent.com/66360846/195597511-65816246-3c7b-4b05-8275-ba038894ede4.png)

![image](https://user-images.githubusercontent.com/66360846/195597536-01f441bd-df8f-4921-9c69-a7545bbb4678.png)

![image](https://user-images.githubusercontent.com/66360846/195597567-e1c836fa-da38-497d-9064-53b7d4f126bd.png)

![image](https://user-images.githubusercontent.com/66360846/195597588-db649710-17be-425e-9321-35cf4a9ea81b.png)

![image](https://user-images.githubusercontent.com/66360846/195597611-36160d7e-b4bd-4893-a7a8-d26bf4ede2fa.png)

![image](https://user-images.githubusercontent.com/66360846/195598122-1041ab68-216d-45a6-adc2-373ea6a1da60.png)

## RESULT:
Feature Generation process and Feature Scaling process is applied to the given data frames sucessfully.
