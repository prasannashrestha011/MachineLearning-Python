import mglearn.datasets
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import mglearn
cb_dataset=load_breast_cancer()
X_train,X_test,y_train,y_test=train_test_split(cb_dataset.data,cb_dataset.target,random_state=0)
#training the model
lr=LinearRegression().fit(X_train,y_train)
rg=Ridge().fit(X_train,y_train)

print("Slope: {}".format(lr.coef_))
print("Intercept: {}".format(lr.intercept_))
## Linear regression
print("Training data set in Linear Regression: {:.2f}".format(lr.score(X_train,y_train)))
print("Test data set in Linear Regression: {:.2f}".format(lr.score(X_test,y_test)))

## Ridge regression(prediction may be less than linear regression because each feature of training data have minimal effect on outcome)
print("Training data in Ridge regression:{:.2f}".format(rg.score(X_train,y_train)))
print("Testing data set in Ridge regression:{:.2f}".format(rg.score(X_test,y_test)))