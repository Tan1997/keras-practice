from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer

model_word2vec = Word2Vec(min_count=1, window=5, size=256, workers=4, batch_words=1000)
model_word2vec.build_vocab(all_doc, progress_per=2000)
model_word2vec.train(all_doc, total_examples=model_word2vec.corpus_count, epochs=5, compute_loss=True, report_delay=60*10)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_title + test_title)