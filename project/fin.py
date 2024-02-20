import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import ast

# Import data
movies = pd.read_csv(r'C:\Users\Юля\Downloads\movies_metadata.csv')
movies['adult'] = movies['adult'].map({'True': True, 'False': False})
movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')

print(movies.head(5))

# Visualize missing values
plt.figure(figsize=(10, 6))
sns.heatmap(movies.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Visualization')
plt.show()

# Clean data
def clean_data(df):
    return df.dropna(subset=['release_date', 'budget', 'revenue'], axis=0).reset_index(drop=True)

movies = clean_data(movies)

# Convert success_or_flop column to numeric values
movies['popularity'] = movies['popularity'].map({'failure': 0, 'hit': 1})

# Convert relevant columns to numeric or datetime types
def convert_types(df, columns):
    for col in columns:
        if col == 'id':
            df[col] = pd.to_numeric(df[col], errors='coerce', downcast='integer')
        elif col in ['budget', 'popularity']:
            df[col] = pd.to_numeric(df[col], errors='coerce', downcast='float')
        else:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

convert_types(movies, ['id', 'budget', 'popularity', 'release_date'])

# Create new columns for year and month of release date
movies['release_year'] = movies['release_date'].dt.year
movies['release_month'] = movies['release_date'].dt.month

# Visualize correlation matrix
correlation_matrix = movies[['budget', 'popularity', 'release_year', 'release_month']].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Matrix')
plt.show()

# Visualize budget and revenue distribution
plt.figure(figsize=(12, 6))
sns.histplot(movies['budget'], bins=30, kde=True, color='blue', label='Budget')
sns.histplot(movies['revenue'], bins=30, kde=True, color='green', label='Revenue')
plt.title('Budget and Revenue Distribution')
plt.xlabel('Amount (in million)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Visualize release year distribution
plt.figure(figsize=(12, 6))
sns.countplot(x='release_year', data=movies, palette='viridis')
plt.title('Release Year Distribution')
plt.xlabel('Release Year')
plt.ylabel('Count')
plt.show()

# Convert genres column to list type
movies['genres'] = movies['genres'].apply(ast.literal_eval)
movies['genres'] = movies['genres'].apply(lambda x: [item for sublist in x for item in sublist])

# Visualize top genres
genres_list = movies['genres'].explode().value_counts().head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=genres_list.values, y=genres_list.index, palette='viridis')
plt.title('Top 10 Genres')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.show()

# Create success_or_flop column based on revenue and budget
movies['success_or_flop'] = np.where(movies['revenue'] > movies['budget'], 1, 0)

# XGBoost model training
X = movies[['budget', 'popularity', 'release_year', 'release_month']]
y = movies['success_or_flop']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()