###
### Dictionary creation with keys UMLS code and value SNOMED CT graph embedding
###

import pickle
import os
import re
from ast import literal_eval
import numpy as np

# Create lists to store files with keys and values from SNOMED embedding directories
keys_file = []
for root, dirs, files in os.walk("/home/felicepaolocolliani/OneDrive/Tesi/Dataset/"
                                 "snomed_embeddings/Keys"):
    for file in files:
        keys_file.append(file)
    keys_file.sort()

embedding_file = []
for root, dirs, files in os.walk("/home/felicepaolocolliani/OneDrive/Tesi/Dataset/"
                                 "snomed_embeddings/Embedding"):
    for file in files:
        embedding_file.append(file)
    embedding_file.sort()

# Get the number of files in the directories
n_file = len(keys_file)

# Create an empty dictionary to store data
big_dict = {}

# Load pre-existing dictionary with only SNOMED entities presents in the embedding
f_snomed = open('SNOMEDtoUMLS_dict_clean_for_embed.obj', 'rb')
dict_snomed = pickle.load(f_snomed)
f_snomed.close()

# Loop over each file
for i in range(n_file):
    # Open and read files containing keys and embeddings
    f_keys = open('/home/felicepaolocolliani/OneDrive/Tesi/Dataset/snomed_embeddings/'
                  'Keys/' + keys_file[i], 'r')
    f_embedding = open('/home/felicepaolocolliani/OneDrive/Tesi/Dataset/'
                       'snomed_embeddings/Embedding/' + embedding_file[i], 'r')
    keys = f_keys.readlines()
    n_keys = len(keys)
    embedding = f_embedding.readlines()
    n_embedding = len(embedding)

    # Convert string representations of arrays into actual numpy arrays
    for k2 in range(n_embedding):
        embedding[k2] = np.array(literal_eval(embedding[k2]))

    # Check if the number of keys and embeddings match
    if n_embedding == n_keys:
        for k1 in range(n_keys):
            # Extract integers keys from the dict_snomed and map them to embedding values
            keys[k1] = int(re.sub(r'[^0-9]', '', keys[k1]))
            if keys[k1] in dict_snomed.keys():
                keys[k1] = dict_snomed[keys[k1]]
                big_dict[keys[k1][0]] = embedding[k1]

# Save the resulting dictionary to a file
f = open("Dict_embed_full.obj", 'wb')
pickle.dump(big_dict, f)
f.close()

