#
# Creation of the dataset that will be used in model_train.py
#
import pickle
import torch

# Import of the text embedding in the MedMention/BC5CDR datasets
dataset_pubtator = 'MM_test_with_SAPBERT_emb.obj'
f = open(dataset_pubtator, 'rb') # full, train, test_dev
data = pickle.load(f)
f.close()

if dataset_pubtator[0:6] == 'BC5CDR':
    f = open('BC5CDRtoUMLS_dict.obj','rb')
    diz_bc5 = pickle.load(f)
    f.close()

# Import of the dictionary with keys Snomed code and values graph embedding (positive)
f = open('Dict_embed_full.obj','rb')
diz_emb = pickle.load(f)
f.close()

# Import of the dictionary with the negative examples
f = open('hard_negative_dict_test_name.obj','rb') # small and full
diz_neg = pickle.load(f)
f.close()

snomed_ent = list(diz_emb.keys()) # Selecting only the snomed entities
x = []  # It will contain the only positive example and the negatives examples
y = []  # It will contain the label [0,1] if the example is negative or positive

for i in range(len(data)): # Cycle for document to process
    print(i)
    for el in data[i].umls_entities:  # Cycle over all the UMLS entity in the file
        if dataset_pubtator[0:6] != 'BC5CDR': # Check on the dataset
            if el.cui in snomed_ent: # Selecting only the snomed entities
                if hasattr(el, 'bert_embedding'): # Selection of the one with a text embedding
                    # appending positive example and label 1
                    x.append(torch.cat((torch.from_numpy(diz_emb[el.cui])[:, None].t(),
                                        el.bert_embedding), 1))
                    y.append(1)
                    # appending negative examples and label 0
                    for neg in diz_neg[el.cui]:
                        x.append(torch.cat((torch.from_numpy(diz_emb[neg])[:, None].t(), el.bert_embedding), 1))
                        y.append(0)
        elif el.cui in diz_bc5.keys():
                if diz_bc5[el.cui][0] in snomed_ent:
                    if hasattr(el,'bert_embedding'):
                        x.append(torch.cat((torch.from_numpy(diz_emb[diz_bc5[el.cui][0]])[:,None].t(),
                                            el.bert_embedding),1))
                        y.append(1)
                        for neg in diz_neg[diz_bc5[el.cui][0]]:
                            x.append(torch.cat((torch.from_numpy(diz_emb[neg])[:,None].t(),
                                                el.bert_embedding),1))
                            y.append(0)


# Saving the data tensors and the labels
f_x = open('x_tensors_full_test.obj','wb') # small, train, test_dev, _NO_RANDOM
pickle.dump(x,f_x)
f_x.close()

f_y = open('y_label_full_test_name.obj','wb') # small, train, test_dev, _NO_RANDOM
pickle.dump(y, f_y)
f_y.close()
