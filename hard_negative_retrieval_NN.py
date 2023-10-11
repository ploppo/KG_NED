#
# Creation of the hard negative dictionary using the KG as ideas
#
import random
from sklearn.neighbors import NearestNeighbors
import pickle

# Define the percentage of the dataset to use
dataset_percentage = 1

# Load dictionary of SNOMED embeddings
f = open('Dict_embed_full.obj', 'rb')
diz = pickle.load(f)
f.close()

# Extract embeddings and corresponding SNOMED entities
embedding = list(diz.values())
snomed_ent = list(diz.keys())
n_entities = len(snomed_ent)

# Initialize a Nearest Neighbors model with a specific number of neighbors
neigh = NearestNeighbors(n_neighbors=int(n_entities*0.10*0.25))
print(int(n_entities*0.10*0.25))  # Print the number of neighbors (e.g., 9379)
neigh.fit(embedding[0:int(n_entities*dataset_percentage)])
print(int(n_entities*dataset_percentage))  # Print the number of entities (e.g., 37518)

# Initialize an empty dictionary with SNOMED entities as keys and their neighbors as values
hard_negative = {}
x = 0

# Loop over each SNOMED entity
for i in range(n_entities):
    # Initialize an empty set for the current entity's hard negatives
    hard_negative[snomed_ent[i]] = set()

    # Find the nearest neighbors of the current entity and add them to the set
    for el in neigh.kneighbors([embedding[i]], 6, return_distance=False)[0][1:]:
        hard_negative[snomed_ent[i]].add(snomed_ent[el])
        # Adding random entity as negative
        while len(hard_negative[snomed_ent[i]]) < 10:
            random_ent = random.choice(snomed_ent)
            if random_ent != snomed_ent[i]:
                hard_negative[snomed_ent[i]].add(random_ent)
    # Convert the set to a list
    hard_negative[snomed_ent[i]] = list(hard_negative[snomed_ent[i]])

# Save the hard negatives dictionary to a file
f = open('hard_negative_dict_full.obj', 'wb')  # Full and small
pickle.dump(hard_negative, f)
f.close()