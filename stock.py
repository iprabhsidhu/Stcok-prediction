import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('tesla.csv')
df.head()
print(df.shape)
print(df.describe())
print(df.info())

plt.figure(figsize=(15,5))
plt.plot(df['Close'])
plt.title('Tesla Close price :', fontsize=15)
plt.ylabel('Price in dollers')
plt.show()

print(df.isnull().sum())

features = ['Open','High', 'Low','Close','Volume']

plt.subplots(figsize=(20,10))

for i, col in enumerate(features):
    plt.subplot(2,3,i+1)
    sb.distplot(df[col])
plt.show()

plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
    plt.subplot(2,3,i+1)
    sb.boxenplot(df[col])
plt.show()

df[['day', 'onth', 'year']] = df['Date'].str.split('-', expand=True)
df['year'] = df['year'].str[:-3].astype(int)

numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
data_grouped = df[numeric_cols].groupby(df['year']).mean()

plt.subplots(figsize=(20,10))

for i, col in enumerate(['Open','High', 'Low','Close','Volume']):
    plt.subplot(2,2,i+1)
    data_grouped[col].plot.bar()
plt.show()

