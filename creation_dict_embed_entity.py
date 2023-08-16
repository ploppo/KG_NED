import pickle
import os
import re
from ast import literal_eval
import numpy as np
###
### produzione del dizionario con chiavi codici UMLS e valori embedding snomedCT
###

keys_file = []
for root, dirs, files in os.walk("/home/felicepaolocolliani/OneDrive/Tesi/Dataset/snomed_embeddings/Keys"):
    for file in files:
        keys_file.append(file)
    keys_file.sort()

embedding_file = []
for root, dirs, files in os.walk("/home/felicepaolocolliani/OneDrive/Tesi/Dataset/snomed_embeddings/Embedding"):
    for file in files:
        embedding_file.append(file)
    embedding_file.sort()

n_file = len(keys_file)
big_dict = {}

f_snomed = open('SNOMEDtoUMLS_dict_clean_for_embed.obj','rb')
dict_snomed = pickle.load(f_snomed)
f_snomed.close()

for i in range(n_file):
    f_keys = open('/home/felicepaolocolliani/OneDrive/Tesi/Dataset/snomed_embeddings/Keys/'+keys_file[i], 'r')
    f_embedding = open('/home/felicepaolocolliani/OneDrive/Tesi/Dataset/snomed_embeddings/Embedding/'+embedding_file[i], 'r')
    keys = f_keys.readlines()
    n_keys = len(keys)
    embedding = f_embedding.readlines()
    n_embedding = len(embedding)
    print(i)
    for k2 in range(n_embedding):
        embedding[k2] = np.array(literal_eval(embedding[k2]))
    if n_embedding == n_keys:
        for k1 in range(n_keys):
            keys[k1] = int(re.sub(r'[^0-9]', '', keys[k1]))
            if keys[k1] in dict_snomed.keys():
                keys[k1] = dict_snomed[keys[k1]]
                big_dict[keys[k1][0]] = embedding[k1]

f = open("Dict_embed_full.obj",'wb')
print('tutto ok')
pickle.dump(big_dict,f)
f.close()
print('fine')
