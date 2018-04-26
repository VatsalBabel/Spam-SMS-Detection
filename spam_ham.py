'''
Used KNeighborsClassifier, SVC, BaggingClassifier, RandomForestClassifier, 
ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier and DecisionTreeClassifier.
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


#Reading the json format dataset using pandas
df = pd.read_json("spam_ham.json")
labels = df['v1'].values
df.drop('v1', axis=1, inplace=True)
features = df.iloc[:,:].values	

#Encoding the features
labelencoder = LabelEncoder()
features[:,0] = labelencoder.fit_transform(features[:,0])
features[:,1] = labelencoder.fit_transform(features[:,1])
features[:,2] = labelencoder.fit_transform(features[:,2])
features[:,3] = labelencoder.fit_transform(features[:,3])

#Encoding the labels
labels = labelencoder.fit_transform(labels)

#Splitting the dataset into train and test set
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

#Training the KNeighborsClassifier on the training data
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(x_train, y_train)
knn.predict(x_test)
knn_accuracy = knn.score(x_test, y_test)
print("KNeighborsClassifier: "+str(knn.score(x_test, y_test)))
#Got 91.86% accuracy

#Training the SVC Classifier on the training data
svc = SVC(kernel='linear')
svc.fit(x_train, y_train)
svc.predict(x_test)
svc_accuracy = svc.score(x_test, y_test)
print("SVC Classifier: "+str(svc.score(x_test, y_test)))
#Got 83.97% accuracy

#Training the BaggingClassifier on the training data
bc = BaggingClassifier()
bc.fit(x_train, y_train)
bc.predict(x_test)
bc_accuracy = bc.score(x_test, y_test)
print("BaggingClassifier: "+str(bc.score(x_test, y_test)))
#Got 90.90% accuracy

#Training the RandomForestClassifier on the training data
rfc = RandomForestClassifier(n_estimators=4)
rfc.fit(x_train, y_train)
rfc.predict(x_test)
rfc_accuracy = rfc.score(x_test, y_test)
print("RandomForestClassifier: "+str(rfc.score(x_test, y_test)))
#Got 91.50% accuracy

#Training the ExtraTreesClassifier on the training data
etc = ExtraTreesClassifier(n_estimators=100)
etc.fit(x_train, y_train)
etc.predict(x_test)
etc_accuracy = etc.score(x_test, y_test)
print("ExtraTreesClassifier: "+str(etc.score(x_test, y_test)))
#Got 91.20% accuracy

#Training the AdaBoostClassifier on the training data
abc = AdaBoostClassifier(n_estimators=100)
abc.fit(x_train, y_train)
abc.predict(x_test)
abc_accuracy = abc.score(x_test, y_test)
print("AdaBoostClassifier: "+str(abc.score(x_test, y_test)))
#Got 86.90% accuracy

#Training the GradientBoostingClassifier on the training data
gbc = GradientBoostingClassifier(n_estimators=100)
gbc.fit(x_train, y_train)
gbc.predict(x_test)
gbc_accuracy = gbc.score(x_test, y_test)
print("GradientBoostingClassifier: "+str(gbc.score(x_test, y_test)))
#Got 90.67% accuracy

#Training the DecisionTreeClassifier on the training data
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
dtc.predict(x_test)
dtc_accuracy = dtc.score(x_test, y_test)
print("DecisionTreeClassifier: "+str(dtc.score(x_test, y_test)))
#Got 90.19% accuracy

plt.bar(1, knn_accuracy*100, label='KNeighborsClassifier')
plt.bar(2, svc_accuracy*100, label='SVC')
plt.bar(3, bc_accuracy*100, label='BaggingClassifier')
plt.bar(4, rfc_accuracy*100, label='RandomForestClassifier')
plt.bar(5, etc_accuracy*100, label='ExtraTreesClassifier')
plt.bar(6, abc_accuracy*100, label='AdaBoostClassifier')
plt.bar(7, gbc_accuracy*100, label='GradientBoostingClassifier')
plt.bar(8, dtc_accuracy*100, label='DecisionTreeClassifier')
plt.bar(17, 0)
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
