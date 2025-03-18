import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the dataset
netflix_data = pd.read_csv("netflix_titles.csv")  # Replace with your file path

# Display the first few rows
print(netflix_data.head())