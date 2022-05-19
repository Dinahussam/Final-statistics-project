#import pandas library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv("diabetes.csv")
df.shape

df.head()

df["gender"]=(df["gender"]=="male").astype(int)
df["diabetes"]=(df["diabetes"]=="Diabetes").astype(int)
df.drop(["patient_number"], axis=1, inplace=True)
df["bmi"] = df["bmi"].str.replace(",",".").astype(float)
df["waist_hip_ratio"] = df["waist_hip_ratio"].str.replace(",",".").astype(float)
df["chol_hdl_ratio"] = df["chol_hdl_ratio"].str.replace(",",".").astype(float)

x=df.drop(['diabetes'], axis=1)
x

df.isnull().sum()

y=df.diabetes
y

df.describe()

ax=sns.displot(data=df,x="age")

ax=sns.countplot(data=df,x="gender",)

ax=sns.countplot(data=df,x="diabetes",)

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Create Decision Tree classifer object
model = DecisionTreeClassifier()

# Train Decision Tree Classifer
model= model.fit(x_train,y_train)

#Predict the response for test dataset
y_pred = model.predict(x_test)

#Evaluation using Accuracy score
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)

y_test.shape

#Evaluation using Confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)

#Evaluation using Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

model.predict([[193,77,49,3.9,19,1,61,119,22.5,118,70,32,38,0.83]])

model.predict([[194,269,38,5.1, 29,1,69,167,24.7,120,70,3,40,0.83]])

!pip install --upgrade scikit-learn==0.20.3

#Import modules for Visualizing Decision trees
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

features=x.columns
features

dot_data = StringIO()
export_graphviz(model, out_file=dot_data,filled=True, rounded=True,special_characters=True,feature_names = features,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes_set.png')
Image(graph.create_png())

import pickle
with open('diabetes.classifier.pickle','wb')as f:
    pickle.dump(model,f)

# Create Decision Tree classifer object
model = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
model = model.fit(x_train,y_train)

#Predict the response for test dataset
y_pred = model.predict(x_test)

# Model Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)


#Better Decision Tree Visualisation
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(model, out_file=dot_data,filled=True, rounded=True,special_characters=True, feature_names = features,class_names=['NO D','D'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('diabetes.png')
Image(graph.create_png())
