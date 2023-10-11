#
# Using the pre-trained model SapBERT to add the embedding to the entities in documents
#
import pickle
import torch
from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence

# Load a pubtator dataset (MM or BC5CDR) from file
dataset_pubtator = 'BC5CDR_Trainset.obj'
f = open(dataset_pubtator, 'rb')
data = pickle.load(f)
f.close()
# Check if the dataset name starts with 'BC5CDR'
if dataset_pubtator[0:6] == 'BC5CDR':
    # Load a dictionary to convert BC5CDR entities to UMLS code
    f = open('BC5CDRtoUMLS_dict.obj', 'rb')
    diz_bc5 = pickle.load(f)
    f.close()

# Load dictionary with SNOMED embeddings
f = open('Dict_embed_full.obj', 'rb')
diz = pickle.load(f)
f.close()

# Initialize a Transformer-based word embedding model with SapBERT
embedding = TransformerWordEmbeddings('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')

# Get a list of SNOMED entities
snomed_ent = list(diz.keys())
start = 0

# Loop over each document in the dataset
for i in range(data.n_documents):
    print(i)
    # Preprocess the document by replacing '-' with spaces, to avoid errors
    data.document_list[i].raw_text = data.document_list[i].raw_text.replace('-', ' ')
    # Get the number of sentences in the document
    n_sent = len(data.document_list[i].sentences)

    sentences = [] # Initialize an empty list to store sentences
    # Create Sentence objects for each sentence in the document and embed them
    for sent_indices in data.document_list[i].sent_start_end_indices:
        sentences.append(Sentence(data.document_list[i].raw_text
                                  [sent_indices[0]:sent_indices[1]]))
        embedding.embed(sentences[:])

    j = 0 # Initialize an index 'j' to keep track of sentences

    # Loop over each UMLS entity in the document
    for el in data.document_list[i].umls_entities:
        if dataset_pubtator[0:6] == 'BC5CDR':
            if el.cui in diz_bc5.keys():
                if any(x in snomed_ent for x in diz_bc5[el.cui]):
                    while el.stop_idx not in range(
                            data.document_list[i].sent_start_end_indices[j][1] + 1):
                        j += 1
                        start = 0
        else:
            while (el.stop_idx not in
                   range(data.document_list[i].sent_start_end_indices[j][1] + 1)):
                j += 1
                start = 0
                for k in range(start, len(sentences[j].tokens)):
                    if j == 0:
                        if sentences[j].tokens[k].start_position == el.start_idx:
                            mean = 1
                            while (sentences[j].tokens[k].end_position!=el.stop_idx and
                                   sentences[j].tokens[k].end_position+1 != el.stop_idx):
                                mean += 1
                                k = k + 1
                            if sentences[j].tokens[k].end_position + 1 == el.stop_idx:
                                print(el.mention_text, sentences[j].tokens[k].form)
                            if mean == 1:
                                el.bert_embedding = (
                                    sentences[j].tokens[k].embedding[:, None].t())
                                start = k
                                break
                            elif mean > 1:
                                mean_embedding = (
                                    sentences[j].tokens[k].embedding[:, None].t())
                                for z in range(1, mean):
                                    mean_embedding = torch.add(sentences[j].tokens[k-z].
                                                              embedding, mean_embedding)
                                el.bert_embedding = torch.div(mean_embedding, mean)
                                start = k
                                break
                    if j > 0:
                        if (sentences[j].tokens[k].start_position +
                                data.document_list[i].sent_start_end_indices[j][0] == el.start_idx):
                            mean = 1
                            while (sentences[j].tokens[k].end_position +
                                   data.document_list[i].sent_start_end_indices[j][0] != el.stop_idx
                                   and sentences[j].tokens[k].end_position +
                                   data.document_list[i].sent_start_end_indices[j][0] + 1 != el.stop_idx):
                                mean += 1
                                k = k + 1
                            if (sentences[j].tokens[k].end_position +
                                    data.document_list[i].sent_start_end_indices[j][0] + 1 == el.stop_idx):
                                print(el.mention_text, sentences[j].tokens[k].form)
                            if mean == 1:
                                el.bert_embedding = (
                                    sentences[j].tokens[k].embedding[:, None].t())
                                start = k
                                break
                            elif mean > 1:
                                mean_embedding = sentences[j].tokens[k].embedding[:, None].t()
                                for z in range(1, mean):
                                    mean_embedding = torch.add(sentences[j].tokens[k-z].embedding,
                                                               mean_embedding)
                                el.bert_embedding = torch.div(mean_embedding, mean)
                                start = k
                                break

# Save the modified dataset with now SapBERT embeddings to a pickle file
f = open('BC5CDR_train_with_SAPBERT_emb.obj', 'wb')
pickle.dump(data, f)
f.close()
