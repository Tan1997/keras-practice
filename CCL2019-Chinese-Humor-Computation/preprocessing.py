import pandas as pd
import jieba

"""获取、分析数据"""
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/development.csv')

print(train_df.info())

"""去停用词+分词"""
def stopwordslist(filepath):
    #使用哈工大停用词词典
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

def seg_sentence(sentence):
    #使用jieba分词对句子进行切分
    sentence_seged = jieba.cut(sentence.strip())
    stopwords = stopwordslist('./stopwords.txt')
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    return outstr

def data_preprocessing(data):
    #中文句子切分结果存放到list
    data_cut = [seg_sentence(sentence) for sentence in data.values]
    return data_cut
