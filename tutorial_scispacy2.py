#
# test di scispacy controllando quante entità vengono predette correttamente dal NER di scispacy
#

import scispacy
import spacy
from scispacy.linking import EntityLinker
from spacy import displacy
import pickle

# importo dataset in pubtator format
dataset_pubtator = 'BC5CDR_small.obj'
f = open('/home/felicepaolocolliani/PycharmProjects/KG_NED/' + dataset_pubtator, 'rb')
data = pickle.load(f)
f.close()

# se il dataset fa parte di BC5CDR scambio le entità del tipo D... con quelle di umls, grazie al dizionario creato
if dataset_pubtator[0:6] == 'BC5CDR':
    f_dict = open("/home/felicepaolocolliani/PycharmProjects/KG_NED/BC5CDRtoUMLS_dict.obj", "rb")
    BC5CDRtoUMLS_dict = pickle.load(f_dict)
    f_dict.close()
    for document in data.document_list:
        for entity in document.umls_entities:
            entity.cui = BC5CDRtoUMLS_dict[entity.cui]

# carico modello NER scispacy
nlp = spacy.load("en_ner_bc5cdr_md")  # en_core_sci_sm #en_ner_craft_md #en_ner_bionlp13cg_md #en_ner_bc5cdr_md
print("add pipe start")  # comando lungo da eseguire, dovrebbe scaricare 1,1 GB di dati (?)
# nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": False, "linker_name": "umls"})
print("add pipe end")

for document in data.document_list:  # per ogni documento pubtator
    text = document.raw_text  # document.abstract non ha titolo
    doc = nlp(text)  # analizzo il testo con il modello scispacy

    # dizionario con chiave codice umls e valori lista di indici start/end nel testo dell'entità in questione
    dict_mention_real = {}
    for el in document.umls_entities:
        if isinstance(el.cui, list): # dataset BC5CDR gives list, MM not, so for coherence put everything list
            dict_mention_real[(el.start_idx, el.stop_idx)] = el.cui
        else:
            dict_mention_real[(el.start_idx, el.stop_idx)] = [el.cui]

    # dizionario con chiave codice umls trovato da scispacy e valori lista di indici start/end trovato da scispacy
    dict_mention_scispasy = {}
    for el in doc.ents:
        if len(el._.kb_ents) > 0:
            # sono tute le entità trovate con le relative probabilità
            dict_mention_scispasy[(el.start_char, el.end_char)] = el._.kb_ents

    count_gold = 0
    # check which one are exactly (gold_mention) found from scispasy
    for key in dict_mention_scispasy.keys():
        if key in dict_mention_real:
            for el in dict_mention_scispasy[key]:
                for umls_code in dict_mention_real[key]:
                    if umls_code in el:
                        if dict_mention_scispasy[key].index(el) > 0:
                            print("choosen element number ", dict_mention_scispasy[key].index(el)+1)
                        count_gold += 1
                        break

    accuracy_gold_mentions = count_gold / len(dict_mention_real)
    print("accuracy gold mention for document ", data.document_list.index(document), "is ", accuracy_gold_mentions)

    # check which one are partially (partial_mention) found from scispasy
    count_partial = 0
    for key1 in dict_mention_scispasy.keys():
        for key2 in dict_mention_real.keys():
            if key1[0] in key2 or key1[1] in key2:
                for el in dict_mention_scispasy[key1]:
                    for umls_code in dict_mention_real[key2]:
                        if umls_code in el:
                            if dict_mention_scispasy[key1].index(el) > 0:
                                print("choosen element number ", dict_mention_scispasy[key1].index(el)+1)
                            count_partial += 1
                            break

    accuracy_partial_mentions = count_partial / len(dict_mention_real)
    print("accuracy partial mention for document ", data.document_list.index(document), "is ", accuracy_partial_mentions)