# Intrusion_Detection_System
## Project Overview
This system is an intrusion detection system by using machine learning algorithms. It is comparing 5 classification models and assessing the performance of the models. Then it utilizes the outperforming model for the prediction of the network (either normal or anomaly).

## Tools
- Excel
- Python - Notebook Jupyter

## Installing and calling the necessary libraries
``` python
!pip install mlxtend
!pip install -U scikit-learn
!pip install -U scikit-learn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree  import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import svm
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import GridSearchCV
from mlxtend.plotting import plot_confusion_matrix
```

## Data uploading
Uploading the Train and the Test dataset from the computer to the notejupyter environment

``` python
Trained_Data = pd.read_csv(r'C:\Users\Hp\Downloads\IDS_FinalProject\Trained_data.csv')
Tested_Data = pd.read_csv(r'C:\Users\Hp\Downloads\IDS_FinalProject\Tested_data.csv')
```

```python
Trained_Data
```
![Trained_Data](https://github.com/user-attachments/assets/b8bd9270-6f45-446b-bff4-d0b4821cb4de)


``` python
Trained_Data
```
![Test_data](https://github.com/user-attachments/assets/2dea682d-0ea8-4eb6-be9f-e39641b5e192)

## Data Processing
``` python
Results = set(Trained_Data['class'].values)
print(Results,end=" ")
```
![anomaly](https://github.com/user-attachments/assets/75b9abc7-8206-4845-af22-79e351b6a432)

## Creation of attack_state column
``` python
Trained_attack = Trained_Data['class'].map(lambda a: 0 if a == 'normal' else 1)
Tested_attack = Tested_Data['class'].map(lambda a: 0 if a == 'normal' else 1)

Trained_Data['attack_state'] = Trained_attack
Tested_Data['attack_state'] = Tested_attack
```

![attack_state](https://github.com/user-attachments/assets/cd886c5d-53cb-43ad-9bff-925cf7a469f0)

## Box Plotting for the Trained and Test Data
Trained Boxplot for checking outliers

``` python
Trained_Data.plot(kind='box', subplots=True, layout=(8, 5), figsize=(20, 40))
plt.show()
```
![Trained_Boxplot](https://github.com/user-attachments/assets/5669229a-52d6-42f6-a13a-d39f959a6f11)

Tested Boxplot for checking outlier
``` python
Tested_Data.plot(kind='box', subplots=True, layout=(8, 5), figsize=(20, 40))
plt.show()
```
![Test_Boxplot](https://github.com/user-attachments/assets/d63db767-f788-4337-b794-8fa727297780)

## Data Encoding
```python
## Data Encoding for Trained_data
Trained_Data = pd.get_dummies(Trained_Data,columns=['protocol_type','service','flag'],prefix="",prefix_sep="")

## Data Encoding for Tested_data
Tested_Data = pd.get_dummies(Tested_Data,columns=['protocol_type','service','flag'],prefix="",prefix_sep="")

LE = LabelEncoder()
attack_LE= LabelEncoder()
Trained_Data['class'] = attack_LE.fit_transform(Trained_Data["class"])
Tested_Data['class'] = attack_LE.fit_transform(Tested_Data["class"])
```

## Data Splitting
```python
### Data Splitting
X_train = Trained_Data.drop('class', axis = 1)
X_train = Trained_Data.drop('attack_state', axis = 1)

X_test = Tested_Data.drop('class', axis = 1)
X_test = Tested_Data.drop('attack_state', axis = 1)


Y_train = Trained_Data['attack_state']
Y_test = Tested_Data['attack_state']
```

## Data Scaling
```python
X_train_train,X_test_train ,Y_train_train,Y_test_train = train_test_split(X_train, Y_train, test_size= 0.25 , random_state=42)
X_train_test,X_test_test,Y_train_test,Y_test_test = train_test_split(X_test, Y_test, test_size= 0.25 , random_state=42)

### Data Scaling
Ro_scaler = RobustScaler()
X_train_train = Ro_scaler.fit_transform(X_train_train) 
X_test_train= Ro_scaler.transform(X_test_train)
X_train_test = Ro_scaler.fit_transform(X_train_test) 
X_test_test= Ro_scaler.transform(X_test_test)
```

## Creation of the Function for the Models/Algorithms
```python
def Evaluate(Model_Name, Model_Abb, X_test, Y_test):
    
    Pred_Value= Model_Abb.predict(X_test)
    Accuracy = metrics.accuracy_score(Y_test,Pred_Value)                      
    Sensitivity = metrics.recall_score(Y_test,Pred_Value)
    Precision = metrics.precision_score(Y_test,Pred_Value)
    F1_score = metrics.f1_score(Y_test,Pred_Value)
    Recall = metrics.recall_score(Y_test,Pred_Value)
    
    print('--------------------------------------------------\n')
    print('The {} Model Accuracy   = {}\n'.format(Model_Name, np.round(Accuracy,3)))
    print('The {} Model Sensitvity = {}\n'.format(Model_Name, np.round(Sensitivity,3)))
    print('The {} Model Precision  = {}\n'.format(Model_Name, np.round(Precision,3)))
    print('The {} Model F1 Score   = {}\n'.format(Model_Name, np.round(F1_score,3)))
    print('The {} Model Recall     = {}\n'.format(Model_Name, np.round(Recall,3)))
    print('--------------------------------------------------\n')
    
    Confusion_Matrix = metrics.confusion_matrix(Y_test, Pred_Value)
    plot_confusion_matrix(Confusion_Matrix,class_names=['Normal', 'Attack'],figsize=(5.55,5), colorbar= "blue")
    #plot_roc_curve(Model_Abb, X_test, Y_test)

def GridSearch(Model_Abb, Parameters, X_train, Y_train):
    Grid = GridSearchCV(estimator=Model_Abb, param_grid= Parameters, cv = 3, n_jobs=-1)
    Grid_Result = Grid.fit(X_train, Y_train)
    Model_Name = Grid_Result.best_estimator_
    
    return (Model_Name)
```
