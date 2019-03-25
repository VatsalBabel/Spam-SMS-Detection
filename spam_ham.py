'''
Used KNeighborsClassifier, SVC, BaggingClassifier, RandomForestClassifier, 
ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier and DecisionTreeClassifier.
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#Reading the json format dataset using pandas
df = pd.read_json("spam_ham.json")
labels = df['v1']
features = df['v2']

#Splitting the dataset into train and test set
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

#Vectorizing the labels
cv = CountVectorizer(stop_words = 'english')
count_train = cv.fit_transform(x_train)
count_test = cv.transform(x_test)

#Training the MultinomialNB on the training data
mb = MultinomialNB()
mb.fit(count_train, y_train)
mb_predicted = mb.predict(count_test)
print("MultinomialNB: " + str(accuracy_score(y_test, mb_predicted)))
#Got 98.27% accuracy

#Training the KNeighborsClassifier on the training data
knn = KNeighborsClassifier()
knn.fit(count_train, y_train)
knn_predicted = knn.predict(count_test)
print("KNeighborsClassifier: " + str(accuracy_score(y_test, knn_predicted)))
#Got 91.45% accuracy

#Training the SVC Classifier on the training data
svc = SVC(kernel='linear')
svc.fit(count_train, y_train)
svc_predicted = svc.predict(count_test)
print("SVC Classifier: " + str(accuracy_score(y_test, svc_predicted)))
#Got 97.97% accuracy

#Training the BaggingClassifier on the training data
bc = BaggingClassifier()
bc.fit(count_train, y_train)
bc_predicted = bc.predict(count_test)
print("BaggingClassifier: " + str(accuracy_score(y_test, bc_predicted)))
#Got 97.30% accuracy

#Training the RandomForestClassifier on the training data
rfc = RandomForestClassifier(n_estimators=4)
rfc.fit(count_train, y_train)
rfc_predicted = rfc.predict(count_test)
print("RandomForestClassifier: " + str(accuracy_score(y_test, rfc_predicted)))
#Got 95.81% accuracy

#Training the ExtraTreesClassifier on the training data
etc = ExtraTreesClassifier(n_estimators=100)
etc.fit(count_train, y_train)
etc_predicted = etc.predict(count_test)
print("ExtraTreesClassifier: " + str(accuracy_score(y_test, etc_predicted)))
#Got 97.97% accuracy

#Training the AdaBoostClassifier on the training data
abc = AdaBoostClassifier(n_estimators=100)
abc.fit(count_train, y_train)
abc_predicted = abc.predict(count_test)
print("AdaBoostClassifier: " + str(accuracy_score(y_test, abc_predicted)))
#Got 97.25% accuracy

#Training the GradientBoostingClassifier on the training data
gbc = GradientBoostingClassifier(n_estimators=100)
gbc.fit(count_train, y_train)
gbc_predicted = gbc.predict(count_test)
print("GradientBoostingClassifier: " + str(accuracy_score(y_test, gbc_predicted)))
#Got 95.87% accuracy

#Training the DecisionTreeClassifier on the training data
dtc = DecisionTreeClassifier()
dtc.fit(count_train, y_train)
dtc_predicted = dtc.predict(count_test)
print("DecisionTreeClassifier: " + str(accuracy_score(y_test, gbc_predicted)))
#Got 95.87% accuracy

plt.bar(1, accuracy_score(y_test, knn_predicted)*100, label='KNeighborsClassifier')
plt.bar(2, accuracy_score(y_test, svc_predicted)*100, label='SVC')
plt.bar(3, accuracy_score(y_test, bc_predicted)*100, label='BaggingClassifier')
plt.bar(4, accuracy_score(y_test, rfc_predicted)*100, label='RandomForestClassifier')
plt.bar(5, accuracy_score(y_test, etc_predicted)*100, label='ExtraTreesClassifier')
plt.bar(6, accuracy_score(y_test, abc_predicted)*100, label='AdaBoostClassifier')
plt.bar(7, accuracy_score(y_test, gbc_predicted)*100, label='GradientBoostingClassifier')
plt.bar(8, accuracy_score(y_test, dtc_predicted)*100, label='DecisionTreeClassifier')
plt.bar(9, accuracy_score(y_test, mb_predicted)*100, label='MultinomialNB')
plt.bar(17, 0)
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
