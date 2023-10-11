#
# Test of the best pre-trained model: choosing between SapBERT or BERT
#
from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence
from torch import nn

# Input embedding model
model = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'  #bert-base-uncased
embedding = TransformerWordEmbeddings(model)

# Cystic fibrosis real examples
sentence1 = Sentence('Cystic fibrosis is a genetic condition. It is caused by a faulty gene that affects the movement '
                     'of salt and water in and out of cells.')
sentence2 = Sentence('Cystic fibrosis is an inherited disorder that causes severe damage to the lungs, digestive '
                     'system and other organs in the body.')
sentence3 = Sentence('The primary cause of morbidity and death in people with cystic fibrosis is progressive lung disease.')
# Fake sentence
sentence4 = Sentence('The vaccine for the new COVID variant cystic fibrosis is going to be released soon.')

sentenceCF = [sentence1, sentence2, sentence3, sentence4]
# Embed words in sentence
for sentence in sentenceCF:
    embedding.embed(sentence)

# Collecting cystic fibrosis embeddings and doing the mean between the two word embedding
cf_embeddings = [(sentence1.tokens[0].embedding + sentence1.tokens[1].embedding)/2,
                 (sentence2.tokens[0].embedding + sentence2.tokens[1].embedding)/2,
                 (sentence3.tokens[10].embedding + sentence3.tokens[11].embedding)/2,
                 (sentence4.tokens[7].embedding + sentence4.tokens[8].embedding)/2]
# Check cosine similarity
print("Cosine similarity for CF, model: "+model)
for el1 in cf_embeddings:
    for el2 in cf_embeddings:
        cos = nn.CosineSimilarity(dim=0)
        print(cos(el1, el2))

# Penicillin real examples
sentence5 = Sentence('Penicillins are a group of antibiotics used to treat a wide range of bacterial infections.')  # 0
sentence6 = Sentence('The penicillins are chemically described as 4-thia-1-azabicyclo (3.2.0) heptanes.')  # 1
sentence7 = Sentence('Phenoxymethylpenicillin is a type of penicillin antibiotic. It is used to treat bacterial'
                     'infections, including ear, chest, throat and skin infections.')  # 5
# Fake sentence
sentence8 = Sentence('The penicillin is a tumor that can evolve in the brain.')  # 6

sentencepenicellin = [sentence5, sentence6, sentence7, sentence8]

# Embed words in sentence
for sentence in sentencepenicellin:
    embedding.embed(sentence)

# Collecting penicillin word embedding
penicillin_embeddings = [sentence5.tokens[0].embedding,
                         sentence6.tokens[1].embedding,
                         sentence7.tokens[5].embedding,
                         sentence8.tokens[1].embedding]
# Check cosine similarity
print("Cosine similarity for Penicillin, model:"+model)
for el1 in penicillin_embeddings:
    for el2 in penicillin_embeddings:
        cos = nn.CosineSimilarity(dim=0)
        print(cos(el1, el2))
