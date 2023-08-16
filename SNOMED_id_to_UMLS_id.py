import pickle
# prendo tutte le entità nel file MRCONSO ridotto
f = open("/home/felicepaolocolliani/OneDrive/Tesi/Dataset/MRCONSO_BC5CDR_SNOMED_red.txt", "r")
identities = set()
SNOMEDtoUMLS_dict = {} # chiave codice 9 numeri SNOMED, valori lista di codici UMLS associati, possono essere diversi
for l in f.readlines():
    line = l.split('|')
    if len(line) == 4:
        print(line)
        if line[0] not in identities:
            SNOMEDtoUMLS_dict.setdefault(line[2], [])
            SNOMEDtoUMLS_dict[line[2]].append(line[0])
            identities.add(line[0])
a = list(identities)
print(len(a))
fnew = open("/home/felicepaolocolliani/OneDrive/PycharmProjects/KG_NED/SNOMEDtoUMLS_dict.obj", "wb")
pickle.dump(SNOMEDtoUMLS_dict, fnew)
fnew.close()