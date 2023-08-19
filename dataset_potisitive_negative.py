import pickle
import torch

f = open('MM_full_with_SAPBERT_emb.obj', 'rb') # small and full
data = pickle.load(f)
f.close()

f = open('Dict_embed_full.obj','rb')
diz_emb = pickle.load(f)
f.close()

f = open('hard_negative_dict_small.obj','rb') # small and full
diz_neg = pickle.load(f)
f.close()

snomed_ent = list(diz_neg.keys())
lista = []
x = []
y = []
w = 0

for i in range(data.n_documents):
    print(i)
    for el in data.document_list[i].umls_entities:
        if el.cui in snomed_ent:
            if hasattr(el,'bert_embedding'):
                lista.append((el.mention_text, el.cui, diz_emb[el.cui], el.bert_embedding, 1))
                x.append(torch.cat((torch.from_numpy(diz_emb[el.cui])[:,None].t(), el.bert_embedding)))
                y.append(1)
                for neg in diz_neg[el.cui]:
                    lista.append((el.mention_text, el.cui, neg, diz_emb[neg], el.bert_embedding, 0))
                    x.append(torch.cat((torch.from_numpy(diz_emb[neg])[:,None].t(), el.bert_embedding)))
                    y.append(0)

f_x = open('x_tensors_small.obj','wb') # need full
pickle.dump(x,f_x)
f_x.close()

f_y = open('y_label_small.obj','wb') # need full
pickle.dump(y, f_y)
f_y.close()

f_list = open('lista_x_y_small.obj','wb') # need full
pickle.dump(lista, f_list)
f_list.close()
