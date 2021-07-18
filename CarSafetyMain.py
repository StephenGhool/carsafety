import sklearn as skl
import sklearn.metrics
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_squared_error,accuracy_score
import pandas as pd

#this lobrary allows us to encoded the data differently than the functions included with sklearn
import category_encoders as ce

#read in the Dataset
total_data = pd.read_csv("car.data",sep=",")
name = pd.read_csv("car.names",sep=";")

#we need to clean the data
LE = LabelEncoder()
encoded_features = ["vhigh","vhigh.1","2","2.1","small","low","unacc"]
total_data[encoded_features] = total_data[encoded_features].apply(LE.fit_transform)

# to remove the zero term such that we can divide without errors
total_data = total_data + 1

#segment the data into X and y
X = total_data.iloc[:,0:6]
y = total_data.iloc[:,6]

#increase the number of features to reduce the probability of underfitting
X["x1x2"] = X.iloc[:,1]/X.iloc[:,3]
X["MainSafe"] = X.iloc[:,1]/X.iloc[:,5]
X["bootsafe"] = X.iloc[:,4]/X.iloc[:,5]
X["PersonsSafe"] = X.iloc[:,3]/X.iloc[:,5]

#Note: the above added featuers gave better performance with the poly features generator

#creation of polynomial features
poly = PolynomialFeatures(2)
X = poly.fit_transform(X)
print(X)

#splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=43)

# #we need to normalize the data
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.fit_transform(X_test)

#need to train the model
model = linear_model.LogisticRegression(multi_class='multinomial', solver = 'lbfgs')
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

#Model Performance
print('Coefficients:',model.coef_)
print("Mean Sq Err: %.2f" % mean_squared_error(y_test,y_pred))
print("accuracy: ", model.score(X_train,y_train))
print("accuracy: ", model.score(X_test,y_test))

print(total_data)