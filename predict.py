from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

cv_tags = CountVectorizer()
vectors = cv_tags.fit_transform(data['tags']).toarray()

cv_crew = CountVectorizer()
vectors1 = cv_crew.fit_transform(data['crew']).toarray()

cv_cast = CountVectorizer()
vectors2 = cv_cast.fit_transform(data['cast']).toarray()

cv_genres = CountVectorizer()
vectors3 = cv_genres.fit_transform(data['genres']).toarray()

similarity = cosine_similarity(vectors)
similarity1 = cosine_similarity(vectors1)
similarity2 = cosine_similarity(vectors2)
similarity3 = cosine_similarity(vectors3)

def recommend(movie):
    movie_index = data[data['title']==movie].index[0]
    distances =  similarity[movie_index]+similarity1[movie_index]+similarity2[movie_index]+similarity3[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True,key=lambda x: x[1])[1:20]
    
    for i in movies_list:
        print(data.iloc[i[0]].title)

recommend('Spectre')