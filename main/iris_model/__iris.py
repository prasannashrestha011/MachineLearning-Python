from sklearn.datasets import load_iris
iris_dataset=load_iris()

#displays all the data keys
print("Keys",iris_dataset.keys())
#for data description
print("description:\n",iris_dataset['DESCR'])

#target names
print("target names:\n",iris_dataset["target_names"])
#features names
print("features names:\n",iris_dataset["feature_names"])
#dataset of first five cols
# datafield:sepal_length,sepal_width,petal_length,petal_width
print("data:\n",iris_dataset["data"][:5])
