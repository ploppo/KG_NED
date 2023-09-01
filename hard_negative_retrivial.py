import random
from sklearn.neighbors import NearestNeighbors
import pickle

dataset_percentage = 1

f = open('Dict_embed_full.obj', 'rb')
diz = pickle.load(f)
f.close()

embedding = list(diz.values())
UMLS_ent = list(diz.keys())
n_entities = len(UMLS_ent)

neigh = NearestNeighbors(n_neighbors=int(n_entities*dataset_percentage*0.25))
print(int(n_entities*dataset_percentage*0.25)) #9379
neigh.fit(embedding[0:int(n_entities*dataset_percentage)])
print(int(n_entities*dataset_percentage)) #37518

hard_negative = {}  # dizionario con chiave UMLS e valore vicini
x = 341000
for i in range(341306,375181):
    if i >= x:
        print(i)
        x = x + 1000
    hard_negative[UMLS_ent[i]] = set()
    for el in neigh.kneighbors([embedding[i]], 6, return_distance=False)[0][1:]:
        hard_negative[UMLS_ent[i]].add(UMLS_ent[el])
    while len(hard_negative[UMLS_ent[i]]) < 10:
        random_ent = random.choice(UMLS_ent)
        if random_ent != UMLS_ent[i]:
            hard_negative[UMLS_ent[i]].add(random_ent)
    hard_negative[UMLS_ent[i]] = list(hard_negative[UMLS_ent[i]])

f = open('hard_negative_dict_full.obj','wb')  #full and small
pickle.dump(hard_negative, f)
f.close()
print('tutto ok')