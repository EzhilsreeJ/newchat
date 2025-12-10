# neural network
## exp1-Introduction to Kaggle and Data preprocessing
# AIM:
To perform Data preprocessing in a data set downloaded from Kaggle
# EQUIPMENTS REQUIRED:
Hardware – PCs Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook
# ALGORITHM:
STEP 1:Importing the libraries
STEP 2:Importing the dataset
STEP 3:Taking care of missing data
STEP 4:Encoding categorical data
STEP 5:Normalizing the data
STEP 6:Splitting the data into test and train
# program
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
data = pd.read_csv("Churn_Modelling.csv")
data
data.head()
X=data.iloc[:,:-1].values
X
y=data.iloc[:,-1].values
y
data.isnull().sum()
data.duplicated()
data.describe()
data = data.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)
X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)
X_train
X_test
print("Lenght of X_test ",len(X_test))
```
## OUTPUT:

### Dataset:

<img width="1277" height="468" alt="image" src="https://github.com/user-attachments/assets/b21fcf7a-f057-4c2e-accd-d4e61b46a4d8" />

### X Values:

<img width="726" height="178" alt="image" src="https://github.com/user-attachments/assets/e2ffd970-6256-43da-97af-d1ddde17c21c" />

### Y Values:

<img width="670" height="60" alt="image" src="https://github.com/user-attachments/assets/99b54178-6c44-43bf-94b0-c8a2d50aacf9" />

### Null Values:

<img width="315" height="339" alt="image" src="https://github.com/user-attachments/assets/b67c1c75-ef32-4107-b885-461f45ec9b55" />

### Duplicated Values:

<img width="372" height="284" alt="image" src="https://github.com/user-attachments/assets/8f1a8de2-5747-4525-a13d-c1493c94ecb8" />

### Description:

<img width="1269" height="346" alt="image" src="https://github.com/user-attachments/assets/073f08d6-98d3-488c-86b5-dae0177541de" />

### Normalized Dataset:

<img width="799" height="599" alt="image" src="https://github.com/user-attachments/assets/85b6c9c4-d4f2-4548-9c6e-8f36c4f48a9d" />

### Training Data:

<img width="766" height="172" alt="image" src="https://github.com/user-attachments/assets/afa3e758-458b-4081-ae46-6b680aa3ab6b" />

### Testing Data:

<img width="819" height="175" alt="image" src="https://github.com/user-attachments/assets/3b5258fb-0bd0-4ab0-bbc4-3e804cc94f2a" />

<img width="222" height="33" alt="image" src="https://github.com/user-attachments/assets/0d9aad6b-e24e-4b7b-b901-f4f3c84a8895" />


## RESULT:

Thus, Implementation of Data Preprocessing is done in python using a data set downloaded from Kaggle.

----------------------------------------------------------------------------------------------------------------------
# exp2 Implementation of Perceptron for Binary Classification

# AIM:
To implement a perceptron for classification using Python<BR>

# EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook
# ALGORITHM:
STEP 1: Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Plot the data to verify the linear separable dataset and consider only two classes<BR>
STEP 4:Convert the data set to scale the data to uniform range by using Feature scaling<BR>
STEP 4:Split the dataset for training and testing<BR>
STEP 5:Define the input vector ‘X’ from the training dataset<BR>
STEP 6:Define the desired output vector ‘Y’ scaled to +1 or -1 for two classes C1 and C2<BR>
STEP 7:Assign Initial Weight vector ‘W’ as 0 as the dimension of ‘X’
STEP 8:Assign the learning rate<BR>
STEP 9:For ‘N ‘ iterations ,do the following:<BR>
        v(i) = w(i)*x(i)<BR>
         
        W (i+i)= W(i) + learning_rate*(y(i)-t(i))*x(i)<BR>
STEP 10:Plot the error for each iteration <BR>
STEP 11:Print the accuracy<BR>
# PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
class Perceptron:
  def __init__(self,learning_rate=0.1):
    self.learning_rate = learning_rate
    self._b = 0.0
    self._w = None
    self.misclassified_samples = []
  def fit(self, x: np.array, y: np.array, n_iter=10):
    self._b = 0.0
    self._w = np.zeros(x.shape[1])
    self.misclassified_samples = []
    for _ in range(n_iter):
      errors = 0
      for xi, yi in zip(x, y):
        update = self.learning_rate * (yi - self.predict(xi))
        self._b += update
        self._w += update * xi
        errors += int(update != 0.0)
      self.misclassified_samples.append(errors)

  def f(self, x: np.array) -> float:
  return np.dot(x, self._w) + self._b

  def predict(self, x: np.array):
  return np.where(self.f(x) >= 0, 1, -1)
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)
print(df.head())
y = df.iloc[:, 4].values
x = df.iloc[:, 0:3].values
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title('Iris data set')
ax.set_xlabel("Sepal length in width (cm)")
ax.set_ylabel("Sepal width in width (cm)")
ax.set_zlabel("Petal length in width (cm)")

ax.scatter(x[:50, 0], x[:50, 1], x[:50, 2], color='red',
         marker='o', s=4, edgecolor='red', label="Iris Setosa")
ax.scatter(x[50:100, 0], x[50:100, 1], x[50:100, 2], color='blue',
         marker='^', s=4, edgecolor='blue', label="Iris Versicolour")
ax.scatter(x[100:150, 0], x[100:150, 1], x[100:150, 2], color='green',
         marker='x', s=4, edgecolor='green', label="Iris Virginica")
plt.legend(loc='upper left')
plt.show()
x = x[0:100, 0:2] 
y = y[0:100]

plt.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='Setosa')
plt.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='x',
          label='Versicolour')
plt.xlabel("Sepal length")
plt.ylabel("Petal length")
plt.legend(loc='upper left')
plt.show()
y = np.where(y == 'Iris-setosa', 1, -1)
x[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
x[:, 1] = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,random_state=0)

classifier = Perceptron(learning_rate=0.01)
classifier.fit(x_train, y_train)
print("accuracy", accuracy_score(classifier.predict(x_test), y_test)*100)
plt.plot(range(1, len(classifier.misclassified_samples) + 1),classifier.misclassified_samples, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Errors')
plt.show()
```

# OUTPUT:
![image](https://github.com/user-attachments/assets/4b344136-3197-4b31-9d18-2d7f96edbb1b)

![image](https://github.com/user-attachments/assets/2b17d9b4-e8e5-4b68-9b98-984137551dd8)


![image](https://github.com/user-attachments/assets/43fed8f6-fe81-483d-ae95-35d98d2718f5)

![image](https://github.com/user-attachments/assets/e7f58abe-88d2-49a0-8e62-35bf2ba0662e)

![image](https://github.com/user-attachments/assets/560114cc-b183-41d5-b8b1-33310a129e3d)

# RESULT:
 Thus, a single layer perceptron model is implemented using python to classify Iris data set.
