import pickle
f = open('MM_full_PT.obj','rb')
data = pickle.load(f)
f.close()

f = open('hard_negative_dict_full.obj','rb')
SNOMEDtoUMLS_dict = pickle.load(f)
f.close()

document_to_remove = []

snomed_values_emb = set(SNOMEDtoUMLS_dict.keys())

#for el in SNOMEDtoUMLS_dict.values():
#       snomed_values_emb.add(el[0])

snomed_values_emb = list(snomed_values_emb)

for i in range(data.n_documents):
    print(i)
    data.document_list[i].snomed_entities = []
    for el in data.document_list[i].umls_entities:
        if el.cui in snomed_values_emb:
            data.document_list[i].snomed_entities.append(el)
    if len(data.document_list[i].snomed_entities) == 0:
        document_to_remove.append(i)

f = open('MM_full_with_snomed_full.obj', 'wb') # file with attribute snomed_entities more
pickle.dump(data,f)
f.close()
