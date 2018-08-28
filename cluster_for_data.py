#! /usr/bin/python

# -*- coding: utf-8 -*-

from numpy import *
import math
import jieba
from glob import glob
import json
from gensim import corpora, models, similarities, matutils

def loadData(filelist):
    Data = []
    i = 0
    for fileName in filelist:
        with open(fileName) as f:
            for line in f:
                #print line
                line = json.loads(line)
                i  += 1
                title = line['parse_title']
                Data.append(title)

    return Data


def getStopWords():
    stopwords = []
    for word in open("stopwords.txt", "r"):
        stopwords.append(word.decode('utf-8').strip())
    return stopwords


def cutContent(content, stopwords):
    #print stopwords
    cutWords = []
    words = jieba.cut(content)
    #print words
    for word in words:
        if word == u' ':
            continue
        if word not in stopwords:
            cutWords.append(word)
            #print unicode(word)
    return cutWords


def getMaxSimilarity(dictTopic, vector):
    maxValue = 0
    maxIndex = -1
    for k,cluster in dictTopic.iteritems():
        oneSimilarity = mean([matutils.cossim(vector, v) for v in cluster])
        if oneSimilarity > maxValue:
            maxValue = oneSimilarity
            maxIndex = k
    return maxIndex, maxValue


def single_pass(corpus, titles, thres):
    dictTopic = {}
    clusterTopic = {}
    numTopic = 0 
    cnt = 0
    for vector,title in zip(corpus,titles): 
        if numTopic == 0:
            dictTopic[numTopic] = []
            dictTopic[numTopic].append(vector)
            clusterTopic[numTopic] = []
            clusterTopic[numTopic].append(title)
            numTopic += 1
    
        else:
            maxIndex, maxValue = getMaxSimilarity(dictTopic, vector)
            
            #join the most similar topic
            if maxValue > thres:
                dictTopic[maxIndex].append(vector)
                clusterTopic[maxIndex].append(title)
            #else create the new topic
            else:
                dictTopic[numTopic] = []
                dictTopic[numTopic].append(vector)
                clusterTopic[numTopic] = []
                clusterTopic[numTopic].append(title)
                numTopic += 1
        cnt += 1
        if cnt % 1000 == 0:
            print "processing {}".format(cnt)
    return dictTopic, clusterTopic


if __name__ == '__main__':
    filelist = glob('../mobilenews/*.dat')
    datMat = loadData(filelist) #4
    print type(datMat)
    stopWords = getStopWords()

    n = len(datMat)
    print 'total records:', n

    cutData = []
    for i in range(n):
        cutData.append(cutContent(datMat[i], stopWords))
    print 'cutData is done'
    #get VSM
    dictionary = corpora.Dictionary(cutData)

    corpus = [dictionary.doc2bow(title) for title in cutData]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    thres = 0.5
    dictTopic, clusterTopic = single_pass(corpus_tfidf, datMat, thres)

    print "num of Topic: {}".format(len(dictTopic))

    for k,v in clusterTopic.items():
        cluster_title = '\t'.join(v).encode('utf-8')
        print "cluster idx:{} --- {}".format(k,cluster_title)

