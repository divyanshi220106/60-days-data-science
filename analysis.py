import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("train.csv")

# Show first 5 rows
print(df.head())

# Dataset info
print(df.info())

# Visualization
plt.figure(figsize=(8,5))
sns.countplot(x='Category', data=df)

plt.title("Products by Category")
plt.show()