# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset, drop unnecessary columns, and encode categorical variables.  
2.Define the features (X) and target variable (y).  
3.Split the data into training and testing sets.  
4.Train the logistic regression model, make predictions, and evaluate using accuracy and other  

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SANTHOSH KUMAR SS
RegisterNumber:  212225230251
*/
```
```python
# 1. Import Required Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 2. Load the Dataset
# NOTE: Change the file path if your CSV is in a different location
data = pd.read_csv("Placement_Data.csv")
# View first 5 rows
print("First 5 rows of the dataset:")
print(data.head())

# 3. Create a Copy and Drop Unwanted Columns
data1 = data.copy()
# Dropping 'sl_no' (serial number) and 'salary' (not needed for predicting placement)
data1 = data1.drop(["sl_no", "salary"], axis=1)
print("\nData after dropping 'sl_no' and 'salary':")
print(data1.head())

# 4. Check for Missing and Duplicate Values
print("\nChecking for missing values (True = missing):")
print(data1.isnull().any())
print("\nNumber of duplicate rows:")
print(data1.duplicated().sum())

# 5. Encode Categorical Variables using LabelEncoder
# Columns that are categorical (object type)
cat_cols = ["gender", "ssc_b", "hsc_b", "hsc_s", 
            "degree_t", "workex", "specialisation", "status"]
le = LabelEncoder()
for col in cat_cols:
    data1[col] = le.fit_transform(data1[col])
print("\nData after Label Encoding:")
print(data1.head())

# 6. Define Features (X) and Target (y)
# X = all columns except 'status'
X = data1.iloc[:, :-1]
# y = 'status' column
y = data1["status"]
print("\nFeatures (X) sample:")
print(X.head())
print("\nTarget (y) sample:")
print(y.head())

# 7. Split the Dataset into Training and Testing Sets
# test_size=0.2 → 20% test data, 80% training data
# random_state=0 → same split every time (for reproducibility)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
print("\nTraining and testing shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# 8. Create and Train the Logistic Regression Model
# solver='liblinear' works well for small datasets
lr = LogisticRegression(solver="liblinear")
# Train the model
lr.fit(X_train, y_train)

# 9. Make Predictions on the Test Set
y_pred = lr.predict(X_test)
print("\nPredicted values (y_pred):")
print(y_pred)

# 10. Evaluate Model Performance
# Accuracy: percentage of correctly predicted labels
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
# Classification Report: precision, recall, F1-score
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 11. Predict Placement for a New Student
new_student = [[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]]
new_prediction = lr.predict(new_student)
print("\nPrediction for new student (0 = Not Placed, 1 = Placed):")
print(new_prediction[0])
```

## Output:
<img width="832" height="648" alt="image" src="https://github.com/user-attachments/assets/a001716b-5d7a-4d8b-bdf0-c166c9d49163" />
<img width="842" height="581" alt="image" src="https://github.com/user-attachments/assets/f2a6233d-8194-4a64-a844-8b2fe3b8b9de" />
<img width="787" height="570" alt="image" src="https://github.com/user-attachments/assets/31a461b0-cc61-4b0b-a670-8cc16fe9c55c" />
<img width="737" height="290" alt="image" src="https://github.com/user-attachments/assets/44ea8cdb-3668-4d00-b940-89e4c2560f96" />




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
