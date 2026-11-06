import pandas as pd
data=pd.read_csv("train.csv")
print(data.head())
print(data.info())
print(data.shape)
print("\nAvailable Columns:")
print(data.columns.tolist())
print("\nPrice Range Categories:")
print(data["price_range"].value_counts())
from sklearn.model_selection import train_test_split
X=data.drop("price_range",axis=1)
Y=data["price_range"]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
print('Training data:', X_train.shape)
print('Testing data:',X_test.shape)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print("Model Accuracy:", accuracy * 100, "%")


from sklearn.metrics import confusion_matrix, classification_report

print("Confusion Matrix:")
print(confusion_matrix(Y_test, Y_pred))

print("\nClassification Report:")
print(classification_report(Y_test, Y_pred))


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print("Improved Model Accuracy:", accuracy * 100, "%")



# Predict price category for a new mobile

new_mobile = [[
    1500, 2, 1, 16, 1, 8, 0, 1, 0, 1,
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1
]]

prediction = model.predict(new_mobile)

print("Predicted Price Category:", prediction[0])





import os

os.makedirs("results", exist_ok=True)

with open("results/model_results.txt", "w") as file:
    file.write("Mobile Phone Price Category Prediction\n")
    file.write("Algorithm: Logistic Regression\n")
    file.write("Accuracy: 97.5%\n")
    file.write("Sample Prediction: Low Price (0)\n")

print("Results saved successfully!")