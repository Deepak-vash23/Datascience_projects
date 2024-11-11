import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("/content/user_behavior_dataset.csv")

le  = LabelEncoder()
df['Device Model'] = le.fit_transform(df["Device Model"])
df['Operating System'] = le.fit_transform(df["Operating System"])
df['Gender'] = le.fit_transform(df["Gender"])

X = df[['Device Model','Operating System','App Usage Time (min/day)','Screen On Time (hours/day)','Battery Drain (mAh/day)','Number of Apps Installed','Data Usage (MB/day)','Age','Gender']].values
Y = df['User Behavior Class'].values

X_train, X_test, Y_train , Y_test = train_test_split(X,Y, test_size=0.2,random_state=0)

#KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)

new_input = np.array([[7,2,350,7.3,1802,66,1054,21,0,]])
Y_pred = knn.predict(new_input)

# confusion_metric = confusion_matrix(Y_test,Y_pred)
# accuracy= accuracy_score(Y_test,Y_pred)
# precision= precision_score(Y_test,Y_pred,average='macro')
# recall = recall_score(Y_test,Y_pred,average='macro')
# f1 = f1_score(Y_test,Y_pred,average='macro')
print(Y_pred)
print(df.head(8))

# print(df.head(10))
# print("Confusion Metrics :" )
# print(confusion_metric)
# print("Accuracy:",accuracy)
# print("Precision:",precision)
# print("F1 Score:",f1)
# print("Recall:",recall)
# print(Y_pred)
# print(X_test)