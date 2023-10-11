import pickle
from full_text_search import full_text
from abbreviation import nlp

# Specify Neo4j database connection details
argv = ['bolt://localhost:7687', 'neo4j', 'honey-judo-tahiti-moses-fossil-8127', 'snomed']

# Initialize a full-text search object with the provided arguments
full_search = full_text(argv=argv)


f = open('Dict_embed_full.obj', 'rb')
diz_embed = pickle.load(f)
f.close()

snomed_entities = list(diz_embed.keys())

documents_test = pickle.load(open('MM_test_with_SAPBERT_emb.obj', 'rb'))

for doc in documents_test:
    doc_abbreviation = nlp(doc.raw_text)
    for entity in doc.umls_entities:
        if entity in snomed_entities:
            if hasattr(entity, 'bert_embedding'): # Selection of the one with a text embedding
                if entity is abbreviation:
                    entity name = take full name

                res = full_search.search(entity name)

                for el in res:
                    concatenate vector (diz_embed, bert_embedding)

                outputs = model
                max outuputs index
                compare diz_embed trovato con umls in doc


