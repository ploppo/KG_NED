import pickle
import torch
from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence

f = open('MM_full_PT.obj', 'rb')
data = pickle.load(f)
f.close()

embedding = TransformerWordEmbeddings('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')  # bert-base-uncased

start = 0

for i in range(data.n_documents):
    print(i)
    n_sent = len(data.document_list[i].sentences)
    sentences = []
    for sent_indices in data.document_list[i].sent_start_end_indices:
        sentences.append(Sentence(data.document_list[i].raw_text[sent_indices[0]:sent_indices[1]]))
        embedding.embed(sentences[:])
    j = 0  # index sentence to consider
    for el in data.document_list[i].umls_entities:
        while el.stop_idx not in range(data.document_list[i].sent_start_end_indices[j][1] + 1):
            j += 1
            start = 0
        for k in range(start, len(sentences[j].tokens)):
            if j == 0:
                if sentences[j].tokens[k].start_position == el.start_idx:
                    mean = 1
                    while sentences[j].tokens[k].end_position != el.stop_idx and sentences[j].tokens[k].end_position + 1 != el.stop_idx:
                        mean += 1
                        k = k + 1
                    if sentences[j].tokens[k].end_position + 1 == el.stop_idx:
                        print(el.mention_text, sentences[j].tokens[k].form)
                    if mean == 1:
                        el.bert_embedding = sentences[j].tokens[k].embedding[:,None].t()
                        start = k
                        break
                    elif mean > 1:
                        mean_embedding = sentences[j].tokens[k].embedding[:,None].t()
                        for z in range(1, mean):
                            mean_embedding = torch.add(sentences[j].tokens[k - z].embedding, mean_embedding)
                        el.bert_embedding = torch.div(mean_embedding, mean)
                        start = k
                        break
            if j > 0:
                if sentences[j].tokens[k].start_position + data.document_list[i].sent_start_end_indices[j][0] == el.start_idx:
                    mean = 1
                    while sentences[j].tokens[k].end_position + data.document_list[i].sent_start_end_indices[j][
                        0] != el.stop_idx and sentences[j].tokens[k].end_position + data.document_list[i].sent_start_end_indices[j][
                        0] + 1 != el.stop_idx:
                        mean += 1
                        k = k + 1
                    if sentences[j].tokens[k].end_position + data.document_list[i].sent_start_end_indices[j][
                        0] + 1 == el.stop_idx:
                        print(el.mention_text, sentences[j].tokens[k].form)
                    if mean == 1:
                        el.bert_embedding = sentences[j].tokens[k].embedding[:,None].t()
                        start = k
                        break
                    elif mean > 1:
                        mean_embedding = sentences[j].tokens[k].embedding[:,None].t()
                        for z in range(1, mean):
                            mean_embedding = torch.add(sentences[j].tokens[k - z].embedding, mean_embedding)
                        el.bert_embedding = torch.div(mean_embedding, mean)
                        start = k
                        break

f = open('MM_full_with_SAPBERT_emb.obj', 'wb')
pickle.dump(data,f)
f.close()