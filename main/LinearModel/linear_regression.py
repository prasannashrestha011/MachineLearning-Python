import mglearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

cb_dataset=load_breast_cancer()
X_train,X_test,y_train,y_test=train_test_split(cb_dataset.data,cb_dataset.target,random_state=0)
#training the model
lr=LinearRegression().fit(X_train,y_train)
rg=Ridge().fit(X_train,y_train)

print("Slope: {}".format(lr.coef_))
print("Intercept: {}".format(lr.intercept_))
## Linear regression
print("Training Set score in Linear Regression: {:.2f}".format(lr.score(X_train,y_train)))
print("Testing Set score in Linear Regression: {:.2f}".format(lr.score(X_test,y_test)))

## Ridge regression(prediction may be less than linear regression because each feature of training data have minimal effect on outcome)
print("Training Set score in Ridge regression:{:.2f}".format(rg.score(X_train,y_train)))
print("Testing Set score in Ridge regression:{:.2f}".format(rg.score(X_test,y_test)))
## prediction score in Ridge with more alpha
ridge10=Ridge(alpha=10).fit(X_train,y_train)
print("Training Set score with high alpha: {:.2f}".format(ridge10.score(X_train,y_train)))
print("Testing Set score with high alpha: {:.2f}".format(ridge10.score(X_test,y_test)))

## prediction score in Ridge with less alpha
ridge01=Ridge(alpha=0.1).fit(X_train,y_train)
print("Training Set score with less alpha: {:.2f}".format(ridge01.score(X_train,y_train)))
print("Testing Set score with less alpha: {:.2f}".format(ridge01.score(X_test,y_test)))

## plotting all the coefficent and intercept
plt.plot(lr.coef_,'s',label="Linear Regression")
plt.plot(rg.coef_,'o',label="Ridge Regression")
plt.plot(ridge01.coef_,'^',label="Ridge Regression with alpha .1")
plt.plot(ridge10.coef_,'v',label="Ridge Regression with alpha 10")
plt.hlines(0,0,len(lr.coef_))
plt.ylim(-25,25)
plt.legend()

plt.show()
