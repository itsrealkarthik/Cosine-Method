import numpy as np
import pandas as pd
import ast
import warnings
import pickle
import nltk
from nltk.stem.porter import PorterStemmer
warnings.filterwarnings(action="ignore")

movie = pd.read_csv('data/tmdb_5000_movies.csv')
credit = pd.read_csv('data/tmdb_5000_credits.csv')

movies = movie.merge(credit, on="title")

movie = movies[['id','title','overview','genres','keywords','cast','crew']]
movie.dropna(axis=0,inplace=True)

def get_genre(value):
    lst = []
    for i in ast.literal_eval(value):
        lst.append(i['name'])
    return lst

movie['genres'] = movie['genres'].apply(get_genre)
movie['keywords'] = movie['keywords'].apply(get_genre)

def get_cast(value):
    lst = []
    counter = 0
    for i in ast.literal_eval(value):
        if counter !=3:
            lst.append(i['name'])
            counter+=1
        else:
            break
    return lst

movie['cast'] = movie['cast'].apply(get_cast)

def get_director(value):
    lst = []
    counter = 0
    for i in ast.literal_eval(value):
        if i['job']=='Director':
            lst.append(i['name'])
            break
    return lst

movie['crew'] = movie['crew'].apply(get_director)

movie['overview'] = movie['overview'].apply(lambda x: x.split())

movie['genres'] = movie['genres'].apply(lambda x: [i.replace(" ","") for i in x])
movie['keywords'] = movie['keywords'].apply(lambda x: [i.replace(" ","") for i in x])
movie['cast'] = movie['cast'].apply(lambda x: [i.replace(" ","") for i in x])
movie['crew'] = movie['crew'].apply(lambda x: [i.replace(" ","") for i in x])

movie['tags']  = movie['overview']+movie['genres']+movie['keywords']+movie['cast']+movie['crew']

data=movie
data['tags']  = data['tags'].apply(lambda x:" ".join(x))
data['crew']  = data['crew'].apply(lambda x:" ".join(x))
data['cast']  = data['cast'].apply(lambda x:" ".join(x))
data['genres']  = data['genres'].apply(lambda x:" ".join(x))

data['tags']  = data['tags'].apply(lambda x:x.lower())
data['crew']  = data['crew'].apply(lambda x:x.lower())
data['genres']  = data['genres'].apply(lambda x:x.lower())
data['cast']  = data['cast'].apply(lambda x:x.lower())

ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

data['tags'] = data['tags'].apply(stem)
data['crew'] = data['crew'].apply(stem)
data['cast'] = data['cast'].apply(stem)
data['genres'] = data['genres'].apply(stem)

with open('data.pkl', 'wb') as f:
    pickle.dump(data, f)
