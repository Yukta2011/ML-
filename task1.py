import pandas as pd



# Load the dataset
df = pd.read_csv("Titanic-Dataset.csv")

# View basic info
print(df.info())
print(df.describe())
print(df.head())

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# One-hot encode 'Sex' and 'Embarked'
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)



import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load your data into df, e.g., from a CSV file:
df = pd.read_csv('Titanic-Dataset.csv')

scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

import seaborn as sns
import matplotlib.pyplot as plt

# Boxplot to see outliers
sns.boxplot(x=df['Fare'])
plt.show()

Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1

# Filter out outliers
df = df[(df['Fare'] >= Q1 - 1.5 * IQR) & (df['Fare'] <= Q3 + 1.5 * IQR)]







