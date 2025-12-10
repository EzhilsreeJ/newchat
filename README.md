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
--------------------------------------------------------------------------------------------------------------------
# exp3 Implementation of MLP for a non-linearly separable data
# Aim
To implement a perceptron for classification using Python
# Algorithm :
Step 1 : Initialize the input patterns for XOR Gate
Step 2: Initialize the desired output of the XOR Gate
Step 3: Initialize the weights for the 2 layer MLP with 2 Hidden neuron and 1 output neuron
Step 3: Repeat the iteration until the losses become constant and minimum
(i) Compute the output using forward pass output
(ii) Compute the error
(iii) Compute the change in weight ‘dw’ by using backward progatation algorithm.
(iv) Modify the weight as per delta rule.
(v) Append the losses in a list
Step 4 : Test for the XOR patterns.

# program
```
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt
x=np.array([[0,0,1,1],[0,1,0,1]])
y=np.array([[0,1,1,0]])
n_x=2
n_y = 1
n_h = 2
m = x.shape[1]
lr = 0.1
np.random.seed(2)
w1 = np.random.rand(n_h,n_x)   # Weight matrix for hidden layer
w2 = np.random.rand(n_y,n_h)   # Weight matrix for output layer
losses = []
def sigmoid(z):
    z= 1/(1+np.exp(-z))
    return z
def forward_prop(w1,w2,x):
    z1 = np.dot(w1,x)
    a1 = sigmoid(z1)
    z2 = np.dot(w2,a1)
    a2 = sigmoid(z2)
    return z1,a1,z2,a2
def back_prop(m,w1,w2,z1,a1,z2,a2,y):
    dz2 = a2-y
    dw2 = np.dot(dz2,a1.T)/m
    dz1 = np.dot(w2.T,dz2) * a1*(1-a1)
    dw1 = np.dot(dz1,x.T)/m
    dw1 = np.reshape(dw1,w1.shape)
    dw2 = np.reshape(dw2,w2.shape)
    return dz2,dw2,dz1,dw1
iterations = 10000
for i in range(iterations):
    z1,a1,z2,a2 = forward_prop(w1,w2,x)
    loss = -(1/m)*np.sum(y*np.log(a2)+(1-y)*np.log(1-a2))
    losses.append(loss)
    da2,dw2,dz1,dw1 = back_prop(m,w1,w2,z1,a1,z2,a2,y)
    w2 = w2-lr*dw2
    w1 = w1-lr*dw1
plt.plot(losses)
plt.xlabel("EPOCHS")
plt.ylabel("Loss value")
def predict(w1,w2,input):
    z1,a1,z2,a2 = forward_prop(w1,w2,test)
    a2 = np.squeeze(a2)
    if a2>=0.5:
        print( [i[0] for i in input], 1)
    else:
        print( [i[0] for i in input], 0)
print('Input',' Output')
test=np.array([[1],[0]])
predict(w1,w2,test)
test=np.array([[1],[1]])
predict(w1,w2,test)
test=np.array([[0],[1]])
predict(w1,w2,test)
test=np.array([[0],[0]])
predict(w1,w2,test)
```
# output:
# Plot of Losses:
<img width="772" height="542" alt="image" src="https://github.com/user-attachments/assets/4f70dd73-0e24-47bd-b18d-d5fab5a9b8fd" />

# Final Output:
<img width="799" height="305" alt="image" src="https://github.com/user-attachments/assets/7c070fa2-d2b6-4fd6-ab75-4eae88d7cb95" />

# result
Thus, XOR classification problem can be solved using MLP in Python 

----------------------------------------------------------------------------------------------------------------------------------
# exp4-IMPLEMENTATION OF MLP WITH BACKPROPAGATION FOR MULTICLASSIFICATION
# AIM :
To implement a Multilayer Perceptron for Multi classification
## ALGORITHM :

1. Import the necessary libraries of python.

2. After that, create a list of attribute names in the dataset and use it in a call to the read_csv() function of the pandas library along with the name of the CSV file containing the dataset.

3. Divide the dataset into two parts. While the first part contains the first four columns that we assign in the variable x. Likewise, the second part contains only the last column that is the class label. Further, assign it to the variable y.

4. Call the train_test_split() function that further divides the dataset into training data and testing data with a testing data size of 20%.
Normalize our dataset. 

5. In order to do that we call the StandardScaler() function. Basically, the StandardScaler() function subtracts the mean from a feature and scales it to the unit variance.

6. Invoke the MLPClassifier() function with appropriate parameters indicating the hidden layer sizes, activation function, and the maximum number of iterations.

7. In order to get the predicted values we call the predict() function on the testing data set.

8. Finally, call the functions confusion_matrix(), and the classification_report() in order to evaluate the performance of our classifier.

## PROGRAM :
```
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
names = ["Class", "Alcohol", "Malic_acid", "Ash", "Magnesium"]
winedata = pd.read_csv(url, names=names, usecols=[0, 1, 2, 3, 5])
print(winedata.head())
x = winedata.iloc[:, 1:]   # features
y = winedata["Class"]      # labels
# Encode labels
le = preprocessing.LabelEncoder()
y_encoded = le.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.25, random_state=42)
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
mlp = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=1000, random_state=42)
mlp.fit(x_train, y_train)
predictions = mlp.predict(x_test)
predicted_classes = le.inverse_transform(predictions)
# Confusion matrix
print(confusion_matrix(y_test, predictions))
# Classification report with proper class labels
print(classification_report(y_test, predictions, target_names=[f"Wine-Class-{c}" for c in le.classes_]))
```

## OUTPUT :

<img width="927" height="403" alt="image" src="https://github.com/user-attachments/assets/164f6720-d241-4a9a-bdc7-72761d8fc77d" />


## RESULT :
Thus, MLP is implemented for multi-classification using python.
----------------------------------------------------------------------------------------------------------------------------
## exp5 Implementation of XOR using RBF
## Aim:
To implement a XOR gate classification using Radial Basis Function Neural Network.
# ALGORITHM:
Step 1: Initialize the input vector for you bit binary data
Step 2: Initialize the centers for two hidden neurons in hidden layer
Step 3: Define the non- linear function for the hidden neurons using Gaussian RBF
Step 4: Initialize the weights for the hidden neuron
Step 5 : Determine the output function as Y=W1*φ1 +W1 *φ2
Step 6: Test the network for accuracy
Step 7: Plot the Input space and Hidden space of RBF NN for XOR classification.
# program:
```
import numpy as np
import matplotlib.pyplot as plt
def gaussian_rbf(x, landmark, gamma=1):
    return np.exp(-gamma * np.linalg.norm(x - landmark)**2)
def end_to_end(X1, X2, ys, mu1, mu2):
    from_1 = [gaussian_rbf(np.array([X1[i], X2[i]]), mu1) for i in range(len(X1))]
    from_2 = [gaussian_rbf(np.array([X1[i], X2[i]]), mu2) for i in range(len(X1))]
    plt.figure(figsize=(13, 5))
    plt.subplot(1, 2, 1)
    plt.scatter((X1[0], X1[3]), (X2[0], X2[3]), label="Class_0")
    plt.scatter((X1[1], X1[2]), (X2[1], X2[2]), label="Class_1")
    plt.xlabel("$X1$", fontsize=15)
    plt.ylabel("$X2$", fontsize=15)
    plt.title("Xor: Linearly Inseparable", fontsize=15)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.scatter(from_1[0], from_2[0], label="Class_0")
    plt.scatter(from_1[1], from_2[1], label="Class_1")
    plt.scatter(from_1[2], from_2[2], label="Class_1")
    plt.scatter(from_1[3], from_2[3], label="Class_0")
    plt.plot([0, 0.95], [0.95, 0], "k--")
    plt.annotate("Seperating hyperplane", xy=(0.4, 0.55), xytext=(0.55, 0.66),
                arrowprops=dict(facecolor='black', shrink=0.05))
    plt.xlabel(f"$mu1$: {(mu1)}", fontsize=15)
    plt.ylabel(f"$mu2$: {(mu2)}", fontsize=15)
    plt.title("Transformed Inputs: Linearly Seperable", fontsize=15)
    plt.legend()
    A = []
    for i, j in zip(from_1, from_2):
        temp = []
        temp.append(i)
        temp.append(j)
        temp.append(1)
        A.append(temp)
    A = np.array(A)
    W = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(ys)
    print(np.round(A.dot(W)))
    print(ys)
    print(f"Weights: {W}")
    return W

def predict_matrix(point, weights):
    gaussian_rbf_0 = gaussian_rbf(point, mu1)
    gaussian_rbf_1 = gaussian_rbf(point, mu2)
    A = np.array([gaussian_rbf_0, gaussian_rbf_1, 1])
    return np.round(A.dot(weights))
# points
x1 = np.array([0, 0, 1, 1])
x2 = np.array([0, 1, 0, 1])
ys = np.array([0, 1, 1, 0])
# centers
mu1 = np.array([0, 1])
mu2 = np.array([1, 0])
w = end_to_end(x1, x2, ys, mu1, mu2)
# testing
print(f"Input:{np.array([0, 0])}, Predicted: {predict_matrix(np.array([0, 0]), w)}")
print(f"Input:{np.array([0, 1])}, Predicted: {predict_matrix(np.array([0, 1]), w)}")
print(f"Input:{np.array([1, 0])}, Predicted: {predict_matrix(np.array([1, 0]), w)}")
print(f"Input:{np.array([1, 1])}, Predicted: {predict_matrix(np.array([1, 1]), w)}")
```
# output
![image](https://github.com/PriyankaAnnadurai/Ex-5--NN/assets/118351569/47320894-3338-47af-bf16-0e5210ce8844)

# Result:
Thus , a Radial Basis Function Neural Network is implemented to classify XOR data.
------------------------------------------------------------------------------------------------------------------
 # exp6 Heart attack prediction using MLP

 # Aim:
To construct a Multi-Layer Perceptron to predict heart attack using Python
# Algorithm:
Step 1:Import the required libraries: numpy, pandas, MLPClassifier, train_test_split, StandardScaler, accuracy_score, and matplotlib.pyplot.

Step 2:Load the heart disease dataset from a file using pd.read_csv().

Step 3:Separate the features and labels from the dataset using data.iloc values for features (X) and data.iloc[:, -1].values for labels (y).

Step 4:Split the dataset into training and testing sets using train_test_split().

Step 5:Normalize the feature data using StandardScaler() to scale the features to have zero mean and unit variance.

Step 6:Create an MLPClassifier model with desired architecture and hyperparameters, such as hidden_layer_sizes, max_iter, and random_state.

Step 7:Train the MLP model on the training data using mlp.fit(X_train, y_train). The model adjusts its weights and biases iteratively to minimize the training loss.

Step 8:Make predictions on the testing set using mlp.predict(X_test).

Step 9:Evaluate the model's accuracy by comparing the predicted labels (y_pred) with the actual labels (y_test) using accuracy_score().

Step 10:Print the accuracy of the model.

Step 11:Plot the error convergence during training using plt.plot() and plt.show().
# program
```import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt
# Load the dataset (assuming it's stored in a file)
data = pd.read_csv('heart.csv')
# Separate features and labels
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values   # Labels
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Normalize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Create and train the MLP model
mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42)
training_loss = mlp.fit(X_train, y_train).loss_curve_
# Make predictions on the testing set
y_pred = mlp.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# Plot the error convergence
plt.plot(training_loss)
plt.title("MLP Training Loss Convergence")
plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.show()
conf_matrix=confusion_matrix(y_test,y_pred)
classification_rep=classification_report(y_test,y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_rep)
```
# output
![image](https://github.com/user-attachments/assets/9a880072-fd0a-4f99-900a-f71248a090c9)
![image](https://github.com/user-attachments/assets/0518b8d3-7420-4ba1-a991-a46c9ccbfaf8)

# Results:
Thus, an ANN with MLP is constructed and trained to predict the heart attack using python.
