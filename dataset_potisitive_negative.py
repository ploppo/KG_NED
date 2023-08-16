import pickle
import torch
from flair.embeddings import TransformerWordEmbeddings
from flair.embeddings import TransformerDocumentEmbeddings
from flair.data import Sentence
from torch import nn

f = open('MM_full_with_snomed_small.obj', 'rb') # small and full
data = pickle.load(f)
f.close()

f = open('Dict_embed_full.obj','rb')
diz_emb = pickle.load(f)
f.close()

f = open('hard_negative_dict_small.obj','rb') # small and full
diz_neg = pickle.load(f)
f.close()


embedding = TransformerWordEmbeddings('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')  # bert-base-uncased

lista = []
x = []
y = []
w = 0

for documento in data.document_list:
    w += 1
    print(w)
    n_sent = len(documento.sentences)
    sentences = []
    for sent_indices in documento.sent_start_end_indices:
        sentences.append(Sentence(documento.raw_text[sent_indices[0]:sent_indices[1]]))
        embedding.embed(sentences[:])
    j = 0 # index sentence to consider
    for el in documento.snomed_entities:
        while el.stop_idx not in range(documento.sent_start_end_indices[j][1]+1):
            j += 1
        for k in range(len(sentences[j].tokens)):
            if j == 0:
                if sentences[j].tokens[k].start_position == el.start_idx:
                    mean = 0
                    while sentences[j].tokens[k].end_position != el.stop_idx:
                        mean += 1
                        k = k+1
                    if mean == 0:
                        lista.append((el.mention_text, el.cui, diz_emb[el.cui], sentences[j].tokens[k].embedding, 1))
                        x.append(torch.cat((torch.from_numpy(diz_emb[el.cui]), sentences[j].tokens[k].embedding)))
                        y.append(1)
                        for neg in diz_neg[el.cui]:
                            lista.append((el.mention_text ,el.cui, neg, diz_emb[el.cui], diz_emb[neg], 0))
                            x.append(torch.cat((torch.from_numpy(diz_emb[el.cui]), torch.from_numpy(diz_emb[neg]))))
                            y.append(0)
                    elif mean > 0:
                        mean_embedding = sentences[j].tokens[k].embedding
                        for z in range(1,mean):
                            torch.add(sentences[j].tokens[k-z].embedding, mean_embedding)
                        mean_embedding = torch.div(mean_embedding,mean)
                        lista.append((el.mention_text, el.cui, diz_emb[el.cui], mean_embedding, 1))
                        x.append(torch.cat((torch.from_numpy(diz_emb[el.cui]), mean_embedding)))
                        y.append(1)
                        for neg in diz_neg[el.cui]:
                            lista.append((el.mention_text, el.cui, neg, diz_emb[el.cui], diz_emb[neg], 0))
                            x.append(torch.cat((torch.from_numpy(diz_emb[el.cui]), torch.from_numpy(diz_emb[neg]))))
                            y.append(0)
            if j > 0:
                if sentences[j].tokens[k].start_position + documento.sent_start_end_indices[j][0] == el.start_idx:
                    mean = 0
                    while sentences[j].tokens[k].end_position + documento.sent_start_end_indices[j][0] != el.stop_idx:
                        mean += 1
                        k = k+1
                    if mean == 0:
                        lista.append((el.mention_text, el.cui, diz_emb[el.cui], sentences[j].tokens[k].embedding, 1))
                        x.append(torch.cat((torch.from_numpy(diz_emb[el.cui]), sentences[j].tokens[k].embedding)))
                        y.append(1)
                        for neg in diz_neg[el.cui]:
                            lista.append((el.mention_text, el.cui, neg, diz_emb[el.cui], diz_emb[neg], 0))
                            x.append(torch.cat((torch.from_numpy(diz_emb[el.cui]), torch.from_numpy(diz_emb[neg]))))
                            y.append(0)
                    elif mean > 0:
                        mean_embedding = sentences[j].tokens[k].embedding
                        for z in range(1, mean):
                            torch.add(sentences[j].tokens[k-z].embedding, mean_embedding)
                        mean_embedding = torch.div(mean_embedding, mean)
                        lista.append((el.mention_text, el.cui, diz_emb[el.cui], mean_embedding, 1))
                        x.append(torch.cat((torch.from_numpy(diz_emb[el.cui]), mean_embedding)))
                        y.append(1)
                        for neg in diz_neg[el.cui]:
                            lista.append((el.mention_text, el.cui,neg, diz_emb[el.cui], diz_emb[neg], 0))
                            x.append(torch.cat((torch.from_numpy(diz_emb[el.cui]), torch.from_numpy(diz_emb[neg]))))
                            y.append(0)

f_x = open('x_tensors.obj','wb') # need full
pickle.dump(x,f_x)
f_x.close()

f_y = open('y_label.obj','wb') # need full
pickle.dump(y, f_y)
f_y.close()

f_list = open('lista_x_y.obj','wb') # need full
pickle.dump(lista, f_list)
f_list.close()
