import os
import sys
from nltk.tokenize.punkt import PunktWordTokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import Imputer
from sklearn import svm
from sklearn import linear_model
import  sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import math
import re

NEGATIVE = "NEGATIVE"
POSITIVE = "POSITIVE"

ROWNUM    = 0
REGION    = 1
TITLE3P   = 2
AUTHOR3P  = 3
PUB3P     = 4
DATE3P    = 5
DESC3P    = 6
TITLEAMZ  = 7
AUTHORAMZ = 8
PUBAMZ    = 9
DATEAMZ   = 10
DESCAMZ   = 11
TAG       = 12

def learn_svm(X,Y):
    clf = svm.SVC(kernel='poly',degree=2,cache_size=2048,C=1)
    
    scores = cross_val_score(clf, X, Y, cv = 5, scoring='recall')
    print("Recall: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print scores
    scores = cross_val_score(clf, X, Y, cv = 5, scoring='precision')
    print("Precision: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print scores
    
    clf.fit(X, Y)
    Y_pred =  clf.predict(X)
    x = metrics.precision_recall_fscore_support(Y,Y_pred)
    print "Precision\tRecall\n0:%0.2f\t%0.2f\n1:%0.2f\t%0.2f\n"%(x[0][0],x[1][0],x[0][1],x[1][1])


def learn_lr(X,Y):
    for w in [1.0]:
        weight  = {1:w,0:1.0}
        print 'weight',w
        #clf = linear_model.LogisticRegression(penalty='l1',class_weight=weight)
        clf = svm.SVC(kernel='poly',degree=2,cache_size=2048,C=1)
    
        scores = cross_val_score(clf, X, Y, cv = 5, scoring='recall')
        print("Recall: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
        print scores

        scores = cross_val_score(clf, X, Y, cv = 5, scoring='precision')
        print("Precision: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
        print scores

        clf.fit(X, Y)
        Y_pred =  clf.predict(X)
        
        Z = zip(Y,Y_pred)

        x = metrics.precision_recall_fscore_support(Y,Y_pred)
        print "Precision\tRecall\n0:%0.2f\t%0.2f\n1:%0.2f\t%0.2f\n"%(x[0][0],x[1][0],x[0][1],x[1][1])
        coef = clf.coef_[0]
        
        wh = open('coefs%s.txt'%w,'w')
        p = 0
        for i in xrange(len(coef)): 
            if coef[i] != 0.0:
                p+=1
        wh.close()

        wh = open('coefs%s.txt'%w,'w')
        p = 0
        for i in xrange(len(coef)): 
            if coef[i] != 0.0:
                p+=1
        wh.close()

        print "total",len(coef),"non zero", p

def writeOut(fname,R,Y):
    pred = zip(R,Y)
    wh = open(fname ,'w')
    wh.write('"ROW_NUMBER","LABEL"\n')
    for (rownum,py) in pred: 
        if py==0:
            wh.write('"%s","NEGATIVE"\n'%rownum)
        else:
            wh.write('"%s","POSITIVE"\n'%rownum)
    


def train_test(X,Y,R,XT,YT):
    weight  = {1:0.5,0:1.0}
    clf = RandomForestClassifier(n_estimators=30)
    clf.fit(X, Y)
    Y_pred =  clf.predict(XT)
        
    Z = zip(YT,Y_pred)

    x = metrics.precision_recall_fscore_support(YT,Y_pred)
    print "Precision\tRecall\n0:%0.2f\t%0.2f\n1:%0.2f\t%0.2f\n"%(x[0][0],x[1][0],x[0][1],x[1][1])
    
    report = metrics.classification_report(YT,Y_pred)
    print report

    writeOut('expected.txt',R,YT)
    writeOut('predicted.txt',R,Y_pred)

    print 'Errors'
    for i in xrange(len(R)):
        if YT[i] != Y_pred[i]:
            print R[i],YT[i],Y_pred[i] 


def length(title):
    return len(title)

def numTokens(title):
    tokenList = PunktWordTokenizer().tokenize(title)
    return len(tokenList)


def parentheses(title):
    if '(' in title:
        return 1.0
    return 0.0

def separator(title):
    if ':' in title or ',' in title or ';' in title or '-' in title:
        return 1.0
    return 0.0

def removeParenthesesSection(title):
    if '(' in title:
        idx = title.find('(')
        return title[:idx]
    return title

def removeSeparator(title):
    for e in [':','-',';',',']:
        if e in title:
            idx = title.find(e)
            return title[:idx]
    return title


def volume(title):
    if 'volume' in title or 'vol.' in title or 'series' in title or 'edition' in title:
        return 1.0
    return 0.0

def years(title):
    if re.search('[1-2]\d\d\d',title):
        return 1.0
    return 0.0

def numbers(title):
    if re.search('\d+',title):
        return 1.0
    return 0.0

def titleFeatures(title):
    return (length(title),numTokens(title),volume(title),separator(title),parentheses(title),years(title),numbers(title))


def getYear(s):
    match =  re.search('[1-2]\d\d\d',s)
    if match != None:
        y = int(match.group(0))
        return y

    match =  re.search('\d{1,2}-\w{2,3}-(\d\d)',s)
    if match != None:
        y = int(match.group(1))
        if y < 14:
            return 2000+y
        else:
            return 1900+y
    return 0000


def dateFeatures(s1,s2):
    return (getYear(s1)-getYear(s2))


def jaccard(t1,t2):
    intersection = 0.0
    for t in t1:
        if t in t2:
            intersection+=1
    if len(t1) == 0.0 and len(t2) == 0.0:
        return 0.0
    return float(intersection)/float(len(t1) + len(t2) - intersection)

def containtment(t1,t2):
    intersection = 0.0
    for t in t1:
        if t in t2:
            intersection+=1
    if (len(t1)) == 0:
        return 0.0
    return float(intersection)/float(len(t1))


def getShingles(title, shingleSize=4):
    title = re.sub("\s+","",title)
    shingles = set()
    for i in xrange(len(title)-shingleSize):
        shingles.add(title[i:i+shingleSize])
    return shingles
        

def createFeatures(src,tgt):
    srcTokenList = PunktWordTokenizer().tokenize(src)
    tgtTokenList = PunktWordTokenizer().tokenize(tgt)
    
    # unigrams
    srcTokens = set(srcTokenList)        
    tgtTokens = set(tgtTokenList)        
    j = jaccard(srcTokens,tgtTokens)
    clr = containtment(srcTokens,tgtTokens)
    crl = containtment(srcTokens,tgtTokens)

    # bigrams
    srcTokens = set(["%s %s"%(srcTokenList[i],srcTokenList[i+1]) for i in xrange(len(srcTokenList)-1)])
    tgtTokens = set(["%s %s"%(tgtTokenList[i],tgtTokenList[i+1]) for i in xrange(len(tgtTokenList)-1)])
    
    bj = jaccard(srcTokens,tgtTokens)
    bclr = containtment(srcTokens,tgtTokens)
    bcrl = containtment(srcTokens,tgtTokens)

    srcShingles = getShingles(src)
    tgtShingles = getShingles(tgt)

    sj = jaccard(srcShingles,tgtShingles)
    sclr = containtment(srcShingles,tgtShingles)
    scrl = containtment(srcShingles,tgtShingles)

    return (j,clr,crl,bj,bclr,bcrl,sj,sclr,scrl)


def poly(v):
    nv = []
    for x in v:
        nv.append(x)
        nv.append(x*x)
    for i in xrange(len(v)):
        for j in xrange(i+1,len(v)):
            nv.append(v[i]*v[j])
    return nv


vectors_train = []
labels_train  = []

rows_test = []
vectors_test = []
labels_test  = []

import sys
import random
random.seed()


bad_data = {}
with open(sys.argv[2],'r') as f:
    for l in f:
        l = l.strip()
        bad_data[l] = True


c= 0
with open(sys.argv[1],'r') as f:
    next(f)
    for l in f: 
        l = l.strip()
        fields = l.split('","')
        

        rownum = fields[ROWNUM][1:]
        if rownum in bad_data:
            continue

        tag = fields[TAG][:-1]
        tag = tag.strip()
        
        b = 1
        if tag == NEGATIVE:
            b = 0

        vector = []
        vector += list(createFeatures(fields[TITLE3P],fields[TITLEAMZ]))
        vector += list(createFeatures(fields[TITLE3P],removeSeparator(fields[TITLEAMZ])))
        vector += list(createFeatures(fields[TITLE3P],removeParenthesesSection(fields[TITLEAMZ])))
        vector += list(createFeatures(fields[TITLEAMZ],removeSeparator(fields[TITLE3P])))
        vector += list(createFeatures(fields[TITLEAMZ],removeParenthesesSection(fields[TITLE3P])))
        vector += list(createFeatures(fields[AUTHOR3P],fields[AUTHORAMZ]))
        vector += list(createFeatures(fields[PUB3P],fields[PUBAMZ]))
        vector += list(createFeatures(fields[DESC3P][:250],fields[DESCAMZ][:250]))
        vector += list(createFeatures(fields[DESC3P][:100],fields[DESCAMZ][:100]))
        vector += list(titleFeatures(fields[TITLEAMZ]))
        vector += list(titleFeatures(fields[TITLE3P]))
        vector.append(dateFeatures(fields[DATE3P],fields[DATEAMZ]))

        if random.random() < 0.8:
            vectors_train.append(vector)
            labels_train.append(b)
        else:
            rows_test.append(fields[ROWNUM][1:])
            vectors_test.append(vector)
            labels_test.append(b)

        c+=1
        if c % 10000 == 0:
            sys.stdout.write('.')
            sys.stdout.flush()
            
print 'Start Training'
train_test(vectors_train, np.asarray(labels_train), rows_test, vectors_test, np.asarray(labels_test))

