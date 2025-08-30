import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

data = {
    'movie_id': [1, 2, 3, 4, 5],
    'title': ["The Matrix", "John Wick", "The Godfather", "Pulp Fiction", "Inception"],
    'genres': ["Action,Sci-Fi", "Action,Thriller", "Crime,Drama", "Crime,Drama", "Action,Sci-Fi,Thriller"],
    'description': [
        "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.",
        "An ex-hitman comes out of retirement to track down the gangsters that killed his dog and took everything from him.",
        "An organized crime dynasty's aging patriarch transfers control of his clandestine empire to his reluctant son.",
        "The lives of two mob hitmen, a boxer, a gangster's wife, and a pair of diner bandits intertwine in four tales of violence and redemption.",
        "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a CEO."
    ]
}

## Convert the dataset into dataframe
df = pd.DataFrame(data)

#Display the dataframe
print("Movie Recommendation System")
print(df)

#Defione a TF-IDF to vectorizer to transform the genre text into vectors
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['genres'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to recommend movies based on cosine similarity
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = df[df['title'] == title].index[0]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 2most similar movies
    sim_scores = sim_scores[1:3]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 5 most similar movie
    return df['title'].iloc[movie_indices]

#Test the recommendation system
movie_title = "The Matrix"
recommended_movies = get_recommendations(movie_title)
print(f"Recommended movies for '{movie_title}':")
for movie in recommended_movies:
    print(f"- {movie}")
