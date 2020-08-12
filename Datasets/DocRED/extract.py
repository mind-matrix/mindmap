import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from copy import deepcopy
from pandas.core.common import flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def preprocess_doc(vertexSet, sents, label, word2id, ner2id, maxlen):
        
    h_type = ner2id[vertexSet[label['h']][0]['type']]
    t_type = ner2id[vertexSet[label['t']][0]['type']]
    
    i_sent = flatten(deepcopy(sents))
    i_sent = [ word2id[word] if word in word2id else word2id['UNK'] for word in i_sent ]
    
    i_h_sent = deepcopy(sents)
    for h in vertexSet[label['h']]:
        for pos in range(h['pos'][0], h['pos'][1]):
            i_h_sent[h['sent_id']][pos] = 'BLANK'
    i_h_sent = list(flatten(i_h_sent))
    i_h_sent = [ word2id[word] if word in word2id else word2id['UNK'] for word in i_h_sent ]
    
    i_t_sent = deepcopy(sents)
    for t in vertexSet[label['t']]:
        for pos in range(t['pos'][0], t['pos'][1]):
            i_t_sent[t['sent_id']][pos] = 'BLANK'
    i_t_sent = list(flatten(i_t_sent))
    i_t_sent = [ word2id[word] if word in word2id else word2id['UNK'] for word in i_t_sent ]
    
    i_sent = pad_sequences([i_sent], maxlen=maxlen, padding='post')[0].tolist()
    i_h_sent = pad_sequences([i_h_sent], maxlen=maxlen, padding='post')[0].tolist()
    i_t_sent = pad_sequences([i_t_sent], maxlen=maxlen, padding='post')[0].tolist()
    
    return i_sent, i_h_sent, i_t_sent, h_type, t_type


def preprocess(file, word2id, ner2id, rel2id, maxlen, save_ds=None):
    with open(file, 'r') as fd:
        docs = json.load(fd)
    with open(word2id, 'r') as fd:
        word2id = json.load(fd)
    with open(ner2id, 'r') as fd:
        ner2id = json.load(fd)
    with open(rel2id, 'r') as fd:
        rel2id = json.load(fd)
        
    dataset = []
    
    for doc in tqdm(docs, total = len(docs)):
        for label in doc['labels']:
            i_sent, i_h_sent, i_t_sent, h_type, t_type = preprocess_doc( doc['vertexSet'], doc['sents'], label, word2id, ner2id, maxlen )
            dataset.append([ i_sent, i_h_sent, i_t_sent, h_type, t_type, rel2id[ label['r'] ] ])
    
    df = pd.DataFrame(dataset, columns=['sents','h_sents','t_sents','h','t','labels'])
    
    df.dropna(inplace=True)
    
    df['labels'] = df['labels'].apply(lambda x: to_categorical(x, 97))
    
    if save_ds is not None:
        df.to_hdf(save_ds, "/docred/%d"%(maxlen))
    
    return [ np.array([ np.array(x) for x in df['sents'].values ]),
            np.array([ np.array(x) for x in df['h_sents'].values ]),
            np.array([ np.array(x) for x in df['t_sents'].values ]),
            np.array(df['h'].values),
            np.array(df['t'].values) ], np.array([ np.array(x) for x in df['labels'].values ])

def from_ds(dataset, maxlen):
    df = pd.read_hdf(dataset, "/docred/%d"%(maxlen))
    return [ np.array([ np.array(x) for x in df['sents'].values ]),
        np.array([ np.array(x) for x in df['h_sents'].values ]),
        np.array([ np.array(x) for x in df['t_sents'].values ]),
        np.array(df['h'].values),
        np.array(df['t'].values) ], np.array([ np.array(x) for x in df['labels'].values ])