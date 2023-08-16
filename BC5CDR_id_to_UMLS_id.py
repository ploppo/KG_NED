import pickle
# prendo tutte le entità nel file MRCONSO ridotto
f = open("/home/felicepaolocolliani/Desktop/Tesi/Dataset/MRCONSO_red.txt", "r")
identities = set()
BC5CDRtoUMLS_dict = {} # chiave codice D.... BC5CDR,valori lista di codici UMLS associati, possono essere diversi
for l in f.readlines():
    line = l.split('|')
    if line[0] not in identities:
        if len(line[1]) > 0: # se le entità umls hanno un corrispettivo che inizia con D le prendo
            if line[1][0] == 'D':
                BC5CDRtoUMLS_dict.setdefault(line[1], [])
                BC5CDRtoUMLS_dict[line[1]].append(line[0])
                identities.add(line[0])
fnew = open("/home/felicepaolocolliani/PycharmProjects/KG_NED/BC5CDRtoUMLS_dict.obj", "wb")
pickle.dump(BC5CDRtoUMLS_dict, fnew)
fnew.close()
