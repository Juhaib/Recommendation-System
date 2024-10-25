import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load the dataset
anime_data = pd.read_csv('anime.csv')

# Remove leading and trailing spaces from column names
anime_data.columns = anime_data.columns.str.strip()

# Print the columns in the DataFrame
print("Columns in dataset:", anime_data.columns.tolist())

# Display the first few rows of the DataFrame
print(anime_data.head(10))

# Handle missing values
anime_data.dropna(inplace=True)

# One-hot encoding of genres
anime_data = anime_data.join(anime_data['genre'].str.get_dummies(sep=', '))

# Convert rating and episodes to numeric and handle errors
anime_data['rating'] = pd.to_numeric(anime_data['rating'], errors='coerce')
anime_data['episodes'] = pd.to_numeric(anime_data['episodes'], errors='coerce')

# Drop any rows that couldn't be converted to numeric
anime_data.dropna(subset=['rating', 'episodes'], inplace=True)

# Normalize numerical features
scaler = MinMaxScaler()
anime_data[['rating', 'episodes']] = scaler.fit_transform(
    anime_data[['rating', 'episodes']]
)

# Select features for similarity
features = anime_data.drop(['anime_id', 'name', 'genre', 'type', 'members'], axis=1, errors='ignore')

# Compute cosine similarity
similarity_matrix = cosine_similarity(features)

# Function to recommend anime based on cosine similarity
def recommend_anime(target_title, num_recommendations=5):
    if target_title not in anime_data['name'].values:
        return "Anime not found."
    
    target_index = anime_data[anime_data['name'] == target_title].index[0]
    similarity_scores = list(enumerate(similarity_matrix[target_index]))
    
    # Sort by similarity score
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Get top recommendations
    top_indices = [i[0] for i in sorted_scores[1:num_recommendations + 1]]
    recommended_anime = anime_data.iloc[top_indices]['name']
    
    return recommended_anime.tolist()

# Example usage of the recommendation function
print(recommend_anime('Kimi no Na wa.'))  # Replace with an actual anime title

# Split the dataset for evaluation (optional)
train_data, test_data = train_test_split(anime_data, test_size=0.2, random_state=42)

# Note: Implement evaluation metrics such as precision, recall, and F1-score as needed.
