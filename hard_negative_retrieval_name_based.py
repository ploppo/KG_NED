import pickle
from full_text_search import full_text

# Specify Neo4j database connection details
argv = ['bolt://localhost:7687', 'neo4j', 'honey-judo-tahiti-moses-fossil-8127', 'snomed']

# Initialize a full-text search object with the provided arguments
full_search = full_text(argv=argv)

# Load dictionary of SNOMED embeddings
f = open('Dict_embed_full.obj', 'rb')
diz = pickle.load(f)
f.close()

# Load dictionary with entities which need the negatives examples
f = open('Dict_name_BC5CDR_test.obj', 'rb')
diz2 = pickle.load(f)
f.close()

# Load dictionary mapping SNOMED codes to UMLS codes
f = open('SNOMEDtoUMLS_dict.obj', 'rb')
snomed_umls_dict = pickle.load(f)
f.close()
keys = snomed_umls_dict.keys()

# Extract embeddings and SNOMED entities from the loaded dictionary
embedding = list(diz.values())
snomed_ent = list(diz.keys())

# Extract entity names and their UMLS codes from the loaded dictionary
entity_train = list(diz2.keys())
n_entities = len(entity_train)

# Initialize a dictionary with UMLS codes as keys and similar UMLS codes as values
hard_negative = {}
x = 0

# Loop over each entity in the training set
for i in range(n_entities):
    print(i)
    hard_negative[entity_train[i]] = []

    # Perform a full-text search using the entity's name and retrieve similar results
    res = full_search.search(diz2[entity_train[i]])

    # Process the search results
    if res is not None:
        for el in res[:min(10,len(res))]:
            if el[1] in keys:
                if snomed_umls_dict[el[1]] in snomed_ent:
                    if snomed_umls_dict[el[1]] != entity_train[i]:
                        hard_negative[entity_train[i]].append(snomed_umls_dict[el[1]])


# Save the hard negatives dictionary
f = open('hard_negative_dict_test_name_BC5CDR.obj', 'wb')
pickle.dump(hard_negative, f)
f.close()
