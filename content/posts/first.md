---
date: '2025-02-13T12:28:23-08:00'
draft: false
title: 'Python to Tabpy - Tableau'
description: Omo Portfolio description
cover:
    image: img/Tabpy image.png # image path/url
    alt: "This" # alt text
    caption: " " # display caption under cover
tags: ["Tableau", "Machine Learning", "TabPy", "Python"]
categories: ["Tech"]

---
 
<div style="text-align: justify; max-width: 700px; margin: auto;">
 This project is one of the data science challenge posted on Kaggle and I decided to jump on it. It is about creating a predictive macine Learning model that predicts which passengers are likely to survive the titanic Shipwreck.
We all know the story behind the famous titanic ship. But just incase you don't know the story behind Titanic, let me give you a brief gist 😉.
Titanic was the largest ship afloat at the time of its completion in 1912. Designed to carry 2,435 passengers plus roughly 900 crew. Passengers were divided into three classes, reflecting the social stratification of the time.
Titanic set sail from Southampton, England, on April 10, 1912, bound for New York City. It stopped briefly in Cherbourg, France, and Queenstown (now Cobh), Ireland, to pick up additional passengers.
Titanic’s wireless operators received multiple iceberg warnings from other ships. Although, the crew was aware of ice in the vicinity, they maintained speed hoping to arrive earlier.
On the night of April 14, 1912, at around 11:40 p.m. ship’s time, despite evasive maneuvers, Titanic scraped along the iceberg’s starboard side causing punctures below the waterline. Five of the ship’s watertight compartments were compromised.
Over about two hours and forty minutes, the bow sank deeper until the stern rose out of the water. An estimated 1500-1517. A disproportionate number of third-class passengers and crew lost their lives compared to first-class passengers, reflecting class-based access to lifeboats. Around 705 people survived. Many were taken aboard the Cunard liner Carpathia, which arrived at the scene after receiving Titanic’s distress signals.
Let's dive into the brief steps of how I handled this project using anaconda jupyter notebook and tableau. We will get the data, explore, process, and build the ML models. We will then depoly the model using TabPy function and see the results in tableau.
You can access the datasource here - [Titanic](https://www.kaggle.com/c/titanic) dataset on Kaggle. The demographic characteristics are the independent variables.
</div>


Step 1: I got data from Kaggle. You can access it here You can download the dataset [here](/files/train.csv).
<br>
I performed some basic data exploration steps to review the data.
```python
train_df.info()
```
![image](/img/info.png#left)
 
 ```python
train_df.describe()
```
![image](/img/describe.png#center)

From this result, while other feature counts are 891, the count of age is 714 depicting missing data. 38% of the passengers out of the training set survived the titanic (mean - 0.38) with ages ranging from 0.4 to 8.
```python
train_df.head(10) 
```
![image](/img/head.png#center)

There is need to convert most features to numeric so the model can process the features accurately.

To take a deeper look at the missing data, I ran this short code

```python
 Calculate missing values per column
missing_total = train_df.isnull().sum()

# Calculate percentage of missing values
missing_percent = (missing_total / train_df.shape[0]) * 100

# Combine into a single DataFrame
missing_data = pd.DataFrame({
    'Total': missing_total,
    '%': missing_percent.round(1)
})

# Sort by total missing values and show top 5
missing_data_sorted = missing_data.sort_values(by='Total', ascending=False)
missing_data_sorted.head(5)
```  
![image](/img/missingdata.png#left)  

To clean the data and fill missing columns, I used the part of the code block **# Fill NAN values in Age with the random numbers generated** in step 2

The result 
![image](/img/Cleaned.png#left)

Step 2: There is need to create and deploy the function that can take the selected parameter values in Tableau as input and return the probability of a person surviving Titanic. I wrote the final Python code as seen below:

```python
# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import tabpy_client
from tabpy_client.client import Client

# Loading the dataset
train_df = pd.read_csv('C:/Users/Documents/portfolio/train.csv')
test_df = pd.read_csv('C:/Users/Documents/portfolio/test.csv')

# Dealing with missing values in the Age variable
data = [train_df, test_df]
for dataset in data:
    mean = train_df["Age"].mean()
    std = train_df["Age"].std()  # Use train_df.std() for consistency
    is_null = dataset["Age"].isnull().sum()
    # Compute random numbers between mean, std, and is_null
    rand_age = np.random.randint(mean - std, mean + std, size=is_null)
    # Fill NAN values in Age with the random numbers generated
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train_df["Age"].astype(int)

# Define age bins and labels
bins = [0, 11, 19, 27, 35, 43, 58, 66, np.inf]
labels = [0, 1, 2, 3, 4, 5, 6, 7]

# Apply pd.cut to categorize Age variable
for dataset in data:
    dataset['Age'] = pd.cut(dataset['Age'], bins=bins, labels=labels, right=False).astype(int)

#Encoding the categorical labels
le = LabelEncoder()
for dataset in data:
    dataset['Sex'] = le.fit_transform(dataset['Sex'])
    dataset['Fare'] = dataset['Fare'].fillna(0)
    
# Train the Random Forest Model
random_forest = RandomForestClassifier()
X_train = train_df[['Age', 'Sex', 'Pclass', 'Fare', 'SibSp', 'Parch']]
Y_train = train_df['Survived']
random_forest.fit(X_train, Y_train)

# Define and Deploy the Function
client = Client('http://localhost:9004/')

def titanic_survival_predictor(_arg1, _arg2, _arg3, _arg4, _arg5, _arg6):
    import pandas as pd
    # Get the new app's data in a dictionary
    row = {'Age': _arg1,
           'Sex': _arg2,
           'Pclass': _arg3,
           'Fare': _arg4,
           'SibSp': _arg5,
           'Parch': _arg6}
    
    # Convert it into a DataFrame
    test_data = pd.DataFrame(data=row, index=[0])
    
    # Predict the survival and death probabilities
    predprob_survival = random_forest.predict_proba(test_data)
    
    # Return only the survival probability
    return [probability[1] for probability in predprob_survival]

client.deploy('titanic_survival_predictor', titanic_survival_predictor, 'Predicts survival probability', override=True)
```
There is need to check the accuracy of the model built. Although, it is not the focus of the post but I will show a brief way of checking and interpreting the results.

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define features and labels
features = ['Age', 'Sex', 'Pclass', 'Fare', 'SibSp', 'Parch']
X = train_df[features]
y = train_df['Survived']

# Split the data (e.g., 80% train, 20% test)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model on training set
random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)

# Predict on validation set
y_pred = random_forest.predict(X_val)

# Evaluate accuracy
acc = accuracy_score(y_val, y_pred)
print(f"Accuracy: {acc:.4f}")

# Optional: Confusion matrix and classification report
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))

print("\nClassification Report:")
print(classification_report(y_val, y_pred))

```

The result is as below:

![image](/img/Accuracy.png#left)

The formula for accuracy is 
$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

TP: True Positives (Survived correctly predicted as survived)  

TN: True Negatives (Did not survive correctly predicted as not survived)  

FP: False Positives  (Survived incorrectly predicted as not survived)

FN: False Negatives  (Did not survive incorrectly predicted as survived)

Now, let's understand the accuracy results.  

The model is **81.6%** accurate which is not perfect but reliable.

To know where the error occured, the confusion matrix function will diagnose our model.  
As seen, 89 True Negative and 16 False Positives mean the model predicted 89 deaths correctly and 16 deaths incorrectly.
The 17 False Negatives and 57 True Positives means the model predicted 17 deaths incorrectly and 57 survivors correctly.  

**To put it plainly, 89 deaths and 57 survivors were correctly predicted while 17 survivors and 16 deaths were inaccurately predicted.**  
There are several ways to perfect the model while avoiding overfitting. I might cover this in another project 🙂.

Step 3: I installed Tabpy by following the instructions on the tableau resources list below. This step is necessary to help tableau understand your script and correctly interprete it.

Follow the instructions in the links below to connect Tableau Desktop and configure the analytics extension
- [Configure analytics extension](https://help.tableau.com/current/pro/desktop/en-us/r_connection_manage.htm)  
- [TabPy Installation](https://tableau.github.io/TabPy/docs/server-install.html)  
- [How To build advanced analytics](https://www.tableau.com/blog/building-advanced-analytics-applications-tabpy?_gl=1*trnz20*_ga*OTA3NDY4OTk5LjE3MDcyNDI5NzM.*_ga_8YLN0SNXVS*MTczNzA2NTIyNi4yNzQuMS4xNzM3MDY1MjQyLjAuMC4w&_ga=2.10878772.183077179.1736784848-907468999.1707242973)  

Resources : 
- [BYOM Tableau guide](https://www.tableau.com/developer/learning/bring-your-own-machine-learning-models-tableau-analytics-extensions-api)  
- [Analytics Extensions settings](https://help.tableau.com/current/server/en-us/config_r_tabpy.htm?_gl=1*1nb1q1a*_ga*OTA3NDY4OTk5LjE3MDcyNDI5NzM.*_ga_8YLN0SNXVS*MTczNzA1NzMzMy4yNzMuMS4xNzM3MDU4NTI5LjAuMC4w)

The icons used were sourced from [flaticons.com](https://www.flaticon.com/)

Step 4: Using jupyter IDE, I launched anaconda prompt after installing all the necessary libraries/packages and ran the command "tabpy" this will start listening to the port as seen below

![image](/img/Anaconda_cmd.png#center)

Step 5: Launch tableau, depending on the version you are using, hover on help, go to Settings and performance → Manage Analytics Extensions connection.. and input parameters as seen below then click the test connection button.

![image](/img/Analytics_ext.png#center)

Step 6: Connect to the trained data source and write the script for the analysis. The tableau script for Death and survival prediction was written as in the image below :

![image](/img/Death.png#center)

![image](/img/Survival.png#center)

The simple dashboard draft can  be accessed [here](/files/Titanic.twbx)

This is also a short display of how the dashboard functions while in use 

<div style="text-align:center;">
  <video controls width="720">
    <source src="/videos/Titanic.mp4" type="video/mp4">
    Functionality of the dashboard
  </video>
</div>

