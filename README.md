# EX-06 Implementation of Decision Tree Classifier Model for Predicting Employee Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import pandas

2.Import Decision tree classifier

3.Fit the data in the model

4.Find the accuracy score. 

## Program:
```
Developed by: KRISHNARAJ D
RegisterNumber: 212222230070
```
```python
import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

plt.figure(figsize=(18,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```
## Output:

![1 ml](https://github.com/KRISHNARAJ-D/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119559695/1fbde3f5-c66b-46b2-918e-dfaa88ab4b4a)
![2 ml](https://github.com/KRISHNARAJ-D/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119559695/420d38c3-965d-473a-baec-96534af35dac)


### Accuracy value
![3 ml](https://github.com/KRISHNARAJ-D/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119559695/48a76de1-335d-4a71-9bcf-21cf12b658ff)

### Predicted value
![4 ml](https://github.com/KRISHNARAJ-D/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119559695/3b803b3c-6d11-4385-ac3c-bf28dff44623)

### Result tree
![5 ml](https://github.com/KRISHNARAJ-D/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119559695/4ed12bd8-6a4a-48f7-8b6c-3e59637f193e)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
