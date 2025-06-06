import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load Titanic dataset
df = pd.read_csv("Titanic-Dataset.csv")

print(df.info())        # Data types and missing values
print(df.describe())    # Mean, std, min, max, etc.
print(df.isnull().sum())  # Total missing values column-wise

# Histogram of Age
df['Age'].plot(kind='hist', bins=30, edgecolor='black')

# Boxplot of Fare
sns.boxplot(x=df['Fare'])
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

# Pairplot (optional but slow for large datasets)
sns.pairplot(df[['Age', 'Fare', 'Survived']])
plt.show()

# Survival by Sex
sns.countplot(x='Survived', hue='Sex', data=df)

# Average age by Pclass
print(df.groupby('Pclass')['Age'].mean())

# Interactive scatter plot
fig = px.scatter(df, x='Age', y='Fare', color='Survived')
fig.show()

