import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("/content/Iris.csv")

#Predicting Species column as target value
X = df[['Id','SepalLengthCm','PetalLengthCm','PetalWidthCm']].values
Y = df['Species'].values

#training_testing_spliting the data into 20% testing
X_train, X_test, Y_train , Y_test = train_test_split(X,Y, test_size=0.2,random_state=0)

#Using KNN algorithm
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(X_train, Y_train)

# Y_pred = knn.predict(X_test)

#Using logistic regression
log = LogisticRegression()
log.fit(X_train,Y_train)
y_pred = log.predict(X_test)

# #To manually use the pridicting data
# pred = np.array([[]])
# y_pred = log.predict(pred)

confussion_metrics =confusion_matrix(Y_test,y_pred)
accuracy=accuracy_score(Y_test,y_pred)
classification_rep = classification_report(Y_test, y_pred)
precision= precision_score(Y_test,y_pred,average='macro')
recall = recall_score(Y_test,y_pred,average='macro')
f1 = f1_score(Y_test,y_pred,average='macro')
print("Confusion Metrics :" )
print(confussion_metrics)
print("Accuracy:",accuracy)
print("Precision:",precision)
print("F1 Score:",f1)
print("Recall:",recall)
# print(Y_pred)
# print(X_test)