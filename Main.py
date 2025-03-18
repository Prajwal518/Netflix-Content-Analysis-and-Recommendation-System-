import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the dataset
netflix_data = pd.read_csv("netflix_titles.csv")  # Replace with your file path

# Display the first few rows
# print(netflix_data.head())

# Check for missing values
print(netflix_data.isnull().sum())

# Fill missing 'cast' and 'country' with 'Unknown'
netflix_data['cast'].fillna("Unknown", inplace=True)
netflix_data['country'].fillna("Unknown", inplace=True)

# Fill missing 'date_added' and 'rating' with the mode
netflix_data['date_added'].fillna(netflix_data['date_added'].mode()[0], inplace=True)
netflix_data['rating'].fillna(netflix_data['rating'].mode()[0], inplace=True)

# Drop rows with missing 'director' (since it's a small percentage)
netflix_data.dropna(subset=['director'], inplace=True)

# Convert 'date_added' to datetime format
netflix_data['date_added'] = pd.to_datetime(netflix_data['date_added'])

# Check the cleaned data
print(netflix_data.head())