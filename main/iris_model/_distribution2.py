from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
cancer=load_breast_cancer()
X_train,X_test,y_train,y_test=train_test_split(cancer.data,cancer.target,stratify=cancer.target,random_state=66)
train_accuracy=[]
test_accuracy=[]
n_neighbors=range(1,11)
for n_neighbor in n_neighbors:
    clf=KNeighborsClassifier(n_neighbors=n_neighbor).fit(X_train,y_train)
    train_accuracy.append(clf.score(X_train,y_train))
    test_accuracy.append(clf.score(X_test,y_test))
plt.plot(n_neighbors,train_accuracy,label="training accuracy")
plt.plot(n_neighbors,train_accuracy,label="Test accuracy")

plt.show()