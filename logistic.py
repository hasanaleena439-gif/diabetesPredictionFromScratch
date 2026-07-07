#building logistic regression model from scratch
#by importing diabetes dataset from kaggle
#importing numpy as np
import numpy as np
#importing pandas as pd
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
class LogisticRegression:
    def __init__(self, learning_rate, n_iterations):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
# fit function to train the model with dataset  
    def fit(self, X, Y):  
        #number of data points in the dataset (number of rows) 
        #number of features in the dataset (number of columns)
        self.m, self.n = X.shape 
        # initializing weights and bias to zero
        self.weights = np.zeros(self.n)
        self.bias = 0 
        self.X = X
        self.Y = Y
        #implementing gradient descent for optimization
        for i in range(self.n_iterations):
            self.update_weights()
    def update_weights(self):
        # y_hat formula sigmoid function
        z = self.X.dot(self.weights) + self.bias
        y_hat = 1 / (1 + np.exp(-z))
        #DERIVATIVE OF LOSS FUNCTION
        dw = (1 / self.m) * np.dot(self.X.T, (y_hat - self.Y))
        db = (1 / self.m) * np.sum(y_hat - self.Y)
        #X.T is the transpose of X
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
        # writing the sigmoid equation in a separate function to use it in the predict function
    def predict(self, X):
        z = X.dot(self.weights) + self.bias
        Y_predicted = 1 / (1 + np.exp(-z))
        Y_predicted = np.where(Y_predicted >= 0.5, 1, 0)
        return Y_predicted
#loading the diabetes dataset from csv file
diabetes_data = pd.read_csv('diabetes.csv')
#printing the first 5 rows of the dataset
print(diabetes_data.head())
#printing the last 5 rows of the dataset
print(diabetes_data.tail())
#printing the shape of the dataset
print(diabetes_data.shape)
#information about the dataset
print(diabetes_data.info())
#checking for null values in the dataset
print(diabetes_data.isnull().sum())    
print(diabetes_data.describe())
print(diabetes_data['Outcome'].value_counts())
print(diabetes_data.groupby('Outcome').mean())
#splitting the dataset into features and target variable
X = diabetes_data.drop('Outcome', axis=1)
Y = diabetes_data['Outcome']
scaler = StandardScaler()
print(scaler.fit_transform(X))
standardized_data = scaler.transform(X)
print(standardized_data)
#splitting the dataset into training and testing sets
X=standardized_data
Y=diabetes_data['Outcome']
print(X)
print(Y)
#splitting the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
print(X.shape)
classifier = LogisticRegression(learning_rate=0.01, n_iterations=1000)
classifier.fit(X_train, Y_train)
#predicting the test set results
X_train_prediction= classifier.predict(X_train)
print(X_train_prediction)
training_accuracy = accuracy_score(Y_train,X_train_prediction)
print("Training accuracy:", training_accuracy)
X_test_prediction= classifier.predict(X_test)
testing_accuracy = accuracy_score(Y_test,X_test_prediction)
print("Testing accuracy:", testing_accuracy)
input_data = (5,166,72,19,175,25.8,0.587,51)
#changing the input data to a numpy array
input_data_as_numpy = np.array(input_data)
#reshaping the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy.reshape(1,-1)
#standardizing the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)
prediction = classifier.predict(std_data)
print(prediction)
if (prediction[0] == 0):
    print("The person is not diabetic")
else:
    print("The person is diabetic")
cm= confusion_matrix(Y_test, X_test_prediction)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Diabetic', 'Diabetic'], yticklabels=['Not Diabetic', 'Diabetic'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()