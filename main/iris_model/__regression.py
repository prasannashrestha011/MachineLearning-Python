import mglearn.datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import mglearn
X,y=mglearn.datasets.make_wave(n_samples=40)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
reg=KNeighborsRegressor(n_neighbors=3)
reg.fit(X_train,y_train)

#prediciting on test data
prediction_result=reg.predict(X_test)
print("Prediction result: {}".format(prediction_result))
print("Accuracy: {:.2f}".format(reg.score(X_test,y_test)))
