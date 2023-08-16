from flair.embeddings import TransformerWordEmbeddings
from flair.embeddings import TransformerDocumentEmbeddings
from flair.data import Sentence
from torch import nn
from numpy import dot
from numpy.linalg import norm


# init embedding
embedding = TransformerWordEmbeddings('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')  # bert-base-uncased

# cambridgeltl/SapBERT-from-PubMedBERT-fulltext
                                                            # cambridgeltl/SapBERT-from-PubMedBERT-fulltext

# # CYSTIC FIBROSIS
# sentence1 = Sentence('Cystic fibrosis is a genetic condition. It is caused by a faulty gene that affects the movement '
#                      'of salt and water in and out of cells. .')
# sentence2 = Sentence('Cystic fibrosis is an inherited disorder that causes severe damage to the lungs, digestive '
#                      'system and other organs in the body.')
# sentence3 = Sentence('There are some newer techniques that allow men with cystic fibrosis to have children. ')
# # fake sentence
# sentence4 = Sentence('Tonight for dinner I will cook cystic fibrosis and pasta')

# Penicillin
sentence5 = Sentence('Penicillins are a group of antibiotics used to treat a wide range of bacterial infections.')  # 0
sentence6 = Sentence('The penicillins are chemically described as 4-thia-1-azabicyclo (3.2.0) heptanes.')  # 1
sentence7 = Sentence('Phenoxymethylpenicillin is a type of penicillin antibiotic. It is used to treat bacterial'
                     'infections, including ear, chest, throat and skin infections.')  # 5
# fake sentence
# sentence8 = Sentence('The construction site in the outskirt penicillin is due to be completed.')  # 6
sentence8 = Sentence('The penicillin is a tumor that can evolve in the brain.')  # 6

# sentenceCF = [sentence1, sentence2, sentence3, sentence4]
sentencepenicellin = [sentence5, sentence6, sentence7, sentence8]
# embed words in sentence
# for sentence in sentenceCF:
#     embedding.embed(sentence)
#
# penicillin_embeddings = [sentence1.tokens[0].embedding,
#                          sentence2.tokens[1].embedding,
#                          sentence3.tokens[5].embedding,
#                          sentence4.tokens[7].embedding]

for sentence in sentencepenicellin:
    embedding.embed(sentence)

penicillin_embeddings = [sentence5.tokens[0].embedding,
                         sentence6.tokens[1].embedding,
                         sentence7.tokens[5].embedding,
                         sentence8.tokens[1].embedding]

for el1 in penicillin_embeddings:
    for el2 in penicillin_embeddings:
        cos = nn.CosineSimilarity(dim=0)
        print(cos(el1, el2))
