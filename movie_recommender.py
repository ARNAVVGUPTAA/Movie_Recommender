import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import numpy as np
import requests

# Initialize stemmer and vectorizer
ps = PorterStemmer()
cv = CountVectorizer(max_features=5000, stop_words='english')

# Load data
data = pd.read_csv("dataframe.csv")

# Define the stem function
def stem(text):
    return " ".join(ps.stem(word) for word in text.split())

# Vectorize the 'tags' column
vectors = cv.fit_transform(data['tags']).toarray()
similarity = cosine_similarity(vectors)

def recommend(movie):
    if movie not in data['title'].values:
        print("Movie not found.")
        return

    movie_index = data[data['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    if not movies_list:
        print("No recommendations found.")
        return

    for i in movies_list:
        try:
            print(data.iloc[i[0]].title)
        except IndexError:
            print("Index out of bounds while recommending.")

def get_movie_data(title):
    API_KEY = '23fd07ad'
    BASE_URL = 'http://www.omdbapi.com/'
    response = requests.get(BASE_URL, params={'t': title, 'apikey': API_KEY})
    movie_data = response.json()

    if movie_data.get('Response') == 'True':
        new_df = pd.DataFrame([{
            'movie_id': movie_data['imdbID'][2:],
            'title': movie_data['Title'],
            'tags': movie_data['Plot'] + ' ' + movie_data['Genre'] + ' ' + movie_data['Director'] + ' ' + movie_data['Actors']
        }])
        new_df['tags'] = new_df['tags'].apply(stem)

        # Append new data to the existing DataFrame
        global data
        data = pd.concat([data, new_df], ignore_index=True)

        # Re-fit the CountVectorizer and update vectors and similarity
        global cv, vectors, similarity
        vectors = cv.fit_transform(data['tags']).toarray()
        similarity = cosine_similarity(vectors)

        recommend(new_df['title'].values[0])
    else:
        print(f"Movie not found: {movie_data.get('Error')}")

def find_closest_match(input_title, titles):
    matches = process.extract(input_title, titles, limit=1)
    if matches:
        return matches[0][0]
    return None

movie = input("Enter the name of a movie: ").lower()

if movie in data['title'].values:
    recommend(movie)
else:
    closest_match = find_closest_match(movie, data['title'].values)
    if closest_match:
        confirm = input(f"Did you mean '{closest_match}'? (yes/no): ").strip().lower()
        if confirm == 'yes' or confirm == 'y':
            recommend(closest_match)
        else:
            print("Data not found in my existing database, fetching it from the internet")
            get_movie_data(movie)
    else:
        print("Data not found in my existing database, fetching it from the internet")
        get_movie_data(movie)

# Save updated data to CSV
data.to_csv("dataframe.csv", index=False)
