from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
iris_dataset=load_iris()
X_train,X_test,y_train,y_test=train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)
print("Training data set numebers :{}".format(X_train.shape))
print("Testing data set numebers: {}".format(X_test.shape))
# instantiating the model by feeding training data set
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
predict_result=knn.predict(X_test)
print("Result: {}".format(predict_result))
## printing model accuracy
__accuracy=knn.score(X_test,y_test)
print(__accuracy)