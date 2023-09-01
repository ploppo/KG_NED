import pickle
import torch

f = open('MM_full_with_SAPBERT_emb_train.obj', 'rb') # full, train, test_dev
data = pickle.load(f)
f.close()

f = open('Dict_embed_full.obj','rb')
diz_emb = pickle.load(f)
f.close()

f = open('hard_negative_dict_full.obj','rb') # small and full
diz_neg = pickle.load(f)
f.close()

snomed_ent = list(diz_neg.keys())
lista = []
x = []
y = []

for i in range(len(data)):
    print(i)
    for el in data[i].umls_entities:
        if el.cui in snomed_ent:
            if hasattr(el,'bert_embedding'):
                lista.append((el.mention_text, el.cui, diz_emb[el.cui], el.bert_embedding, 1))
                x.append(torch.cat((torch.from_numpy(diz_emb[el.cui])[:,None].t(), el.bert_embedding),1))
                y.append(1)
                for neg in diz_neg[el.cui]:
                    lista.append((el.mention_text, el.cui, neg, diz_emb[neg], el.bert_embedding, 0))
                    x.append(torch.cat((torch.from_numpy(diz_emb[neg])[:,None].t(), el.bert_embedding),1))
                    y.append(0)

f_x = open('x_tensors_full_train.obj','wb') # small, train, test_dev, need full di tutto
pickle.dump(x,f_x)
f_x.close()

f_y = open('y_label_full_train.obj','wb') # need full
pickle.dump(y, f_y)
f_y.close()

f_list = open('lista_x_y_full_train.obj','wb') # need full
pickle.dump(lista, f_list)
f_list.close()
