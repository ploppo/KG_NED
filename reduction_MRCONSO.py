#
# file che permette di ottenere un file più piccolo e quindi più leggero con tutti i codici UMLS e BC5CDR e nome
# scientifico sulla stessa riga, eliminando informazioni che non servono alla ricerca
#

f1 = open("/home/felicepaolocolliani/OneDrive/Tesi/Dataset/2023AA-full/MRCONSO1.RRF", 'r')
f2 = open("/home/felicepaolocolliani/OneDrive/Tesi/Dataset/2023AA-full/MRCONSO2.RRF", 'r')
fnew = open("/home/felicepaolocolliani/OneDrive/MRCONSO_BC5CDR_SNOMED_red.txt", "w")
lines1 = f1.readlines()
lines2 = f2.readlines()
a = set()
for line in lines1:
    list_split = line.split("|")
    # print(line)
    if len(list_split) >= 14:
        if list_split[11][0:6] == "SNOMED":
            fnew.write(list_split[0] + "|" + list_split[10] + "|" + list_split[9] + "|" + list_split[14] + '\n')
            a.add(list_split[9])
        else:
            fnew.write(list_split[0] + "|" + list_split[10] + "|" + list_split[14] + '\n')
for line in lines2:
    list_split = line.split("|")
    # print(line)
    if len(list_split) >= 14:
        if list_split[11][0:6] == "SNOMED":
            fnew.write(list_split[0] + "|" + list_split[10] + "|" + list_split[9] + "|" + list_split[14] + '\n')
            a.add(list_split[9])
        else:
            fnew.write(list_split[0] + "|" + list_split[10] + "|" + list_split[14] + '\n')
# print(list_split[0]+"|"+list_split[10]+"|"+list_split[14])
a = list(a)
print(len(a))
