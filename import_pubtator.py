from pubtatortool import PubTatorCorpus
import pickle
train_corpus = PubTatorCorpus(['/home/felicepaolocolliani/OneDrive/Tesi/Dataset/PubTator_format/BC5CDR/CDR.Corpus.v010516/CDR_Full.txt'])
f = open('BC5CDR_full.obj', 'wb')
pickle.dump(train_corpus, f)
f.close()

##### creazione test train dataset
data = pickle.load(open('MM_full_with_SAPBERT_emb.obj', 'rb'))

test = open(
    '/home/felicepaolocolliani/OneDrive/Tesi/Dataset/PubTator_format/MedMentions/full/data/corpus_pubtator_pmids_test.txt',
    'rb')
test_pmids = []
for line in test.readlines():
    test_pmids.append(int(line))
test.close()

test_doc = []
for el in data.document_list:
    if int(el.pmid) in test_pmids:
        test_doc.append(el)