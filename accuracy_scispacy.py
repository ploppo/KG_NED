#
# Scispacy NED quality check: understanding the accuracy of their model
#

import spacy
import pickle

# Import pubtator format dataset (MM or BC5CDR)
dataset_pubtator = 'BC5CDR_small.obj'
f = open('/home/felicepaolocolliani/OneDrive/PycharmProjects/KG_NED/' + dataset_pubtator, 'rb')
data = pickle.load(f)
f.close()

# If the dataset is from BC5CDR, exchange entity starting with "D" with the UMLS ones through the dict
if dataset_pubtator[0:6] == 'BC5CDR':
    f_dict = open("/home/felicepaolocolliani/OneDrive/PycharmProjects/KG_NED/BC5CDRtoUMLS_dict.obj", "rb")
    BC5CDRtoUMLS_dict = pickle.load(f_dict)
    f_dict.close()
    for document in data.document_list:
        for entity in document.umls_entities:
            if entity.cui[0] == 'D':
                entity.cui = BC5CDRtoUMLS_dict[entity.cui[0:7]]

# Loading NER model scispacy choosing between: en_core_sci_md / en_core_sci_sm / en_core_sci_scibert /en_core_sci_lg
#                                              en_ner_craft_md / en_ner_bionlp13cg_md / en_ner_bc5cdr_md
nlp = spacy.load("en_core_sci_lg")
print("add pipe start")  # comando lungo da eseguire, dovrebbe scaricare 1,1 GB di dati
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": False, "linker_name": "umls"})
print("add pipe end")

# Lists to store gold and partial accuracy
accuracy_gold_list = []
accuracy_partial_gold_list = []

ndocument = 0
# Looping on each pubtator document
for document in data.document_list:
    ndocument += 1
    print(ndocument)
    text = document.raw_text  # Document full text
    doc = nlp(text)  # Text embedding analysis with scispacy model

    # Creating dictionary with values correct UMLS code and keys start/end indexes of the mention enity text
    dict_mention_real = {}
    for el in document.umls_entities:
        if isinstance(el.cui, list):  # dataset BC5CDR gives list, MM not, so for coherence put everything list
            dict_mention_real[(el.start_idx, el.stop_idx)] = el.cui
        else:
            dict_mention_real[(el.start_idx, el.stop_idx)] = [el.cui]

    # Creating dictionaty with values UMLS 5 codes found by scispacy e keys list of start/end indexes
    dict_mention_scispasy = {}
    for el in doc.ents:
        if len(el._.umls_ents) > 0:
            # sono tute le entità trovate con le relative probabilità
            dict_mention_scispasy[(el.start_char, el.end_char)] = el._.umls_ents

    count_gold = 0
    # Check which one are exactly (gold_mention) found from scispasy
    for key in dict_mention_scispasy.keys():
        if key in dict_mention_real:
            for el in dict_mention_scispasy[key]:
                for umls_code in dict_mention_real[key]: # Check for the first top 5 candidates
                    if umls_code in el:
                        count_gold += 1
                        break

    accuracy_gold_mentions = count_gold / len(dict_mention_real)
    accuracy_gold_list.append(accuracy_gold_mentions)

    # Check which one are partially (partial_mention) found from scispasy
    count_partial = 0
    for key1 in dict_mention_scispasy.keys():
        for key2 in dict_mention_real.keys():
            if key1[0] in key2 or key1[1] in key2:
                for el in dict_mention_scispasy[key1]:
                    for umls_code in dict_mention_real[key2]:
                        if umls_code in el:
                            count_partial += 1
                            break

    accuracy_partial_mentions = count_partial / len(dict_mention_real)
    accuracy_partial_gold_list.append(accuracy_partial_mentions)

# Storing the values in a file
faccuracy_gold = open('Accuracy_gold_MM_testset_sci_lg.obj', 'wb')
pickle.dump(accuracy_gold_list, faccuracy_gold)
faccuracy_gold.close()

faccuracy_partial_gold = open('Accuracy_partial_gold_MM_testset_sci_lg.obj', 'wb')
pickle.dump(accuracy_partial_gold_list, faccuracy_partial_gold)
faccuracy_partial_gold.close()
