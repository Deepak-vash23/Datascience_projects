
import csv
import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

nf = pd.read_csv("/content/diabetes.csv")
print(nf)
X = nf[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']].values
Y = nf['Outcome'].values
print(X)
print(Y)

logreg = LogisticRegression()
X_train , X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
logreg.fit(X_train,Y_train)
Y_pred = logreg.predict(X_test)
print(Y_test)
print(Y_pred)