from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd
import mglearn
import matplotlib.pyplot as plt
iris_dataset=load_iris()

#displays all the data keys
print("Keys",iris_dataset.keys())
#for data description
print("description:\n",iris_dataset['DESCR'])
#features names
print("features names:\n",iris_dataset["feature_names"])

#dataset of first five cols
# datafield:sepal_length,sepal_width,petal_length,petal_width
print("data:\n",iris_dataset["data"][:5])

#data shape
print("data->shape",iris_dataset["data"].shape)


#target names
print("target names:\n",iris_dataset["target_names"])
#data target
# 0=setosa  versicolor=1 virginica=2
print("data target {}".format(iris_dataset["target"]))
X_train,X_test,y_train,y_test=train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)
print("X train data->{}".format(X_train.shape))
print("X test data->{}".format(X_test.shape))

iris_dataframe=pd.DataFrame(X_train,columns=iris_dataset.feature_names)
grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
 hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
plt.show()
