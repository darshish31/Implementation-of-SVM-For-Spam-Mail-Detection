# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1. Start the Program.

Step 2. Import the necessary packages.

Step 3. Read the given csv file and display the few contents of the data.

Step 4. Assign the features for x and y respectively.

Step 5. Split the x and y sets into train and test sets.

Step 6. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.

Step 7. Find the accuracy of the model.

Step 8. End the Program.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: DARSHINI B
RegisterNumber:  212224230051
*/

import chardet, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn import metrics

# Detect encoding
with open("C:\\Users\\admin\\Desktop\\SEM 4 T1\\ML CSV\\spam.csv", 'rb') as f:
 print(chardet.detect(f.read(100000)))
# Load data
data = pd.read_csv("C:\\Users\\admin\\Desktop\\SEM 4 T1\\ML CSV\\spam.csv", encoding='windows-1252')
print(data.head())
print(data.info())
print(data.isnull().sum())
# Split data
x = data['v1'].values   # Labels
y = data['v2'].values   # Messages
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Text vectorization
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

# Train & predict
model = SVC()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Predictions:", y_pred)
# Accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

## Output:
<img width="777" height="715" alt="image" src="https://github.com/user-attachments/assets/567b524f-c384-44db-85a8-afd76c8c3897" />


![image](https://github.com/user-attachments/assets/e859b002-b90c-4ba3-bbd2-3d2acc62e9d7)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
