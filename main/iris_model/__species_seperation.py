from sklearn.datasets import load_iris
import pandas as pd

iris_dataset=load_iris()
iris_dataframe=pd.DataFrame(iris_dataset.data,columns=iris_dataset.feature_names)


#created a new col species where we classify different irish flower using the the target_names array which will be indexed by it labels target
iris_dataframe["species"]=iris_dataset.target_names[iris_dataset.target]
print(iris_dataframe)
# seperating flowers features based on species
setosa_df=iris_dataframe[iris_dataframe["species"]=="setosa"]
versicolor_df=iris_dataframe[iris_dataframe["species"]=="versicolor"]
virginica_df=iris_dataframe[iris_dataframe["species"]=="virginica"]
print("Setosa class:")
print(setosa_df.head())

print("Versicolor class:")
print(versicolor_df.head())

print("Virginica class:")
print(virginica_df.head())