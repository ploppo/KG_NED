from pubtatortool import PubTatorCorpus
import pickle
train_corpus = PubTatorCorpus(['/home/felicepaolocolliani/OneDrive/Tesi/Dataset/PubTator_format/BC5CDR/CDR.Corpus.v010516/CDR_Full.txt'])
f = open('BC5CDR_full.obj', 'wb')
pickle.dump(train_corpus, f)
f.close()

