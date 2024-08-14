import pandas as pd
import ast
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import pickle
from fuzzywuzzy import process
import random

model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')

# Read recipe interactions and raw recipe data
interactions = pd.read_csv('./RAW_interactions100.csv')
recipes = pd.read_csv('./RAW_first100.csv')

# Merge recipe information and interactions data
merged_data = pd.merge(recipes, interactions, right_on='recipe_id', left_on='id', how='inner')

merged_data.head(10)

# Combine recipe steps into one string
merged_data['combined_recipe'] = merged_data['steps'].apply(lambda x: " ".join(ast.literal_eval(x)))

# Embed the recipes using the Sentence Transformer model
recipe_embeddings = model.encode(merged_data['combined_recipe'])
pickle.dump(recipe_embeddings, open("recipe_embeddings.pickle", 'wb'))

# Load the recipe embeddings from the pickle file
recipe_embeddings = pickle.load(open("recipe_embeddings.pickle", 'rb'))
recipe_embeddings_df = pd.DataFrame(recipe_embeddings)

# Calculate cosine similarities between recipes
cosine_similarities = cosine_similarity(recipe_embeddings)
#print('Pairwise cosine similarities:\n {}\n'.format(cosine_similarities))

# Convert the cosine similarity matrix to a DataFrame
cosine_sim_df = pd.DataFrame(cosine_similarities)
pickle.dump(cosine_sim_df, open('cosine_similarities.pickle', 'wb'))

# Load the cosine similarity DataFrame from the pickle file
cosine_sim_df = pickle.load(open('cosine_similarities.pickle', 'rb'))
data_similarity = cosine_sim_df.unstack().reset_index()
data_similarity.columns = ['recipe1', 'recipe2', 'cosine_similarity']

# Filter out self-similarities (cosine similarity of 1)
data_similarity = data_similarity[data_similarity['cosine_similarity'] < 0.9999]

# Map recipe IDs to recipe names
recipe_dict = {j: i for j, i in enumerate(merged_data['name'])}
data_similarity['recipe1_name'] = data_similarity['recipe1'].map(recipe_dict)
data_similarity['recipe2_name'] = data_similarity['recipe2'].map(recipe_dict)

# Rank recipes based on cosine similarity within each recipe1 group
data_similarity['similarity_rank'] = data_similarity.groupby(['recipe1'])['cosine_similarity'].rank("dense", ascending=False)

# Filter out only the top 5 similar recipes for each recipe1
data_similarity = data_similarity[data_similarity['similarity_rank'] <= 5].reset_index(drop=True)

# Save the final similarity data to a CSV file
data_similarity.to_csv('recipe_similarity.csv')

# Display the top 10 rows of the final similarity data
data_similarity.head(10)



def find_similar_recipes(user_input):
    # try to find exact recipe in data
    if user_input in data_similarity['recipe1_name'].values:
        user_input_data = data_similarity[data_similarity['recipe1_name'] == user_input].copy()
    else:
        # if not an exact match, find the closest match
        closest_match = process.extractOne(user_input, data_similarity['recipe1_name'].unique())
        if closest_match:
            user_input_data = data_similarity[data_similarity['recipe1_name'] == closest_match[0]].copy()
        else:
            user_input_data = pd.DataFrame()  # no match found

    # check if user_input_data is empty
    if user_input_data.empty:
        # no close match found, recommend random recipes or default popular ones
        recommendations = random.sample(list(data_similarity['recipe2_name'].unique()), 3)
    else:
        # sort the data by similarity rank and get top 3 recommendations
        user_input_data.sort_values(inplace=True, by=['similarity_rank'])
        recommendations = [recipe_name for j, recipe_name in enumerate(user_input_data['recipe2_name'].unique()) if j < 3]

    return recommendations
