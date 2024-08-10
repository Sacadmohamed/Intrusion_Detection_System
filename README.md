# Intrusion_Detection_System
## Project Overview
This system is an intrusion detection system by using machine learning algorithms. It is comparing 5 classification models and assessing the performance of the models. Then it utilizes the outperforming model for the prediction of the network (either normal or anomaly).

# Tools
- Excel
- Python - Notebook Jupyter

# Installing and calling the necessary libraries
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

# Data uploading
Uploading the Train and the Test dataset from the computer to the notejupyter environment

``` python
Trained_Data = pd.read_csv(r'C:\Users\Hp\Downloads\IDS_FinalProject\Trained_data.csv')
Tested_Data = pd.read_csv(r'C:\Users\Hp\Downloads\IDS_FinalProject\Tested_data.csv')
```

```python
Trained_Data
```
![Trained_Data](https://github.com/user-attachments/assets/b8bd9270-6f45-446b-bff4-d0b4821cb4de)
