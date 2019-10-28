from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


def TF_IDF(sentence_cut):
    """
    #输入已经切分之后的句子,转换为词频矩阵
    vectorizer = CountVectorizer()
    #统计每个词语的tf-idf权值
    transformer = TfidfVectorizer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(sentence_cut)
    """
    vectorizer = TfidfVectorizer(min_df=1, binary=False, decode_error='ignore')
    vectorizer = vectorizer.fit_transform(sentence_cut).toarray()
    return vectorizer
