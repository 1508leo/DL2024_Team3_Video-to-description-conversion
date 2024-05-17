import os
import numpy as np
import pickle

from scipy.interpolate import interp1d

from common.config import get_vocab_config
from common.logger import logger
from backend.utils import caption_tokenize
from backend.videohandler import VideoHandler

# Read
GLOVE_FILE = get_vocab_config()['GLOVE_FILE']
# GLOVE path (dictionary)

# Read or Write if not exists
WORD_EMBEDDED_CACHE = get_vocab_config()['WORD_EMBEDDED_CACHE']
VOCAB_FILE = get_vocab_config()['VOCAB_FILE']
# WORD_EMBEDDED_CACHE and VOCAB_FILE path(dictionary)

def loadFromPickleIfExists(fname):
		# check whether the file is exist
    if not os.path.exists(fname):
        print("Not loading pickle object from %s, file not found." % fname)
        return None
    try: # file exist
        with open(fname,'rb') as f:  # 'rb' -> read binary
            data = pickle.load(f)  # read
            print("Loading pickle object from %s done." % fname)
            return data
    except Exception as e:  # if there is any accident
        print("Exception in loading pickle object from %s: %s" % (fname, e))
    return None

class Vocab:
    OUTDIM_EMB = 300
    WORD_MIN_FREQ = 5
    VOCAB_SIZE = 9448
    CAPTION_LEN = 15
    
    def __init__(self, data, train_ids):
        # data = dict(id => captions)
        self.specialWords = dict() # build dictionary
        self.specialWords['START'] = '>'
        self.specialWords['END'] = '<'
        self.specialWords['NONE'] = '?!?'
        self.specialWords['EXTRA'] = '___'

        freshWordEmbedding = self.loadWordEmbedding(GLOVE_FILE) 
        for word,enc in self.specialWords.items():
            assert enc in self.wordEmbedding.keys() 
        self.buildVocab(data, train_ids, freshWordEmbedding) 

    def loadWordEmbedding(self, glove_file):
        self.wordEmbedding = loadFromPickleIfExists(WORD_EMBEDDED_CACHE) # load data
        if self.wordEmbedding:      # if the word is recorded
            print("Embedding Loaded")
            return False
        else:
            self.wordEmbedding = dict() # dictionary
            with open(glove_file, 'r') as f:
                for i,line in enumerate(f):  # traverse the file
                    tokens = line.split()  # seperate word
                    tokens = [tok.__str__() for tok in tokens]  # 轉成string
                    word = tokens[0]  # get word
                    self.wordEmbedding[word] = np.asarray(tokens[1:], dtype='float32') # translate
            minVal = float('inf')  # positive infinite
            maxVal = -minVal  # negative infinite

            for v in self.wordEmbedding.values():
                for x in v:
                    minVal = min(minVal,x) # choose min
                    maxVal = max(maxVal,x) # choose max
            mapper = interp1d([minVal,maxVal],[-1,1])
            
            for w in self.wordEmbedding:
                self.wordEmbedding[w] = mapper(self.wordEmbedding[w]) # mapping
            print("Cross Check")
            print(self.wordEmbedding['good']) # check
            self.saveEmbedding() # Save
            return True
        
    def saveEmbedding(self):
        with open(WORD_EMBEDDED_CACHE, 'wb') as f:
            pickle.dump(self.wordEmbedding,f)
            
    def buildVocab(self, data, train_ids, trimEmbedding):
        self.ind2word = loadFromPickleIfExists(VOCAB_FILE) # load 詞彙表
        if not self.ind2word:  # doesn't exist
            x = {}  # empty dictionary，record the number of words
            allWords = set()    # ensure only record the word one time
            for w in self.wordEmbedding.keys():
                allWords.add(w)   # add every words
            
            for _id,captions in data.items(): 
                if _id not in train_ids:
                    continue
                for cap in captions:
                    for w in caption_tokenize(cap): 
                        if w not in allWords:
                            continue
                        if w not in x.keys():  # first time
                            x[w]=1
                        else:
                            x[w]+=1
            assert 'tshirt' not in x.keys()
            assert 'tshirt' not in allWords
        
            self.ind2word = []
            for w,enc in self.specialWords.items():  
                self.ind2word.append(enc)
            self.ind2word.extend([w for w in x.keys() if x[w]>=Vocab.WORD_MIN_FREQ])  
            with open(VOCAB_FILE,'wb') as f:  
                pickle.dump(self.ind2word,f)
        
        self.word2ind = dict()  
        for i,w in enumerate(self.ind2word):  
            self.word2ind[w]=i
        assert 'tshirt' not in self.wordEmbedding.keys()
        assert 'tshirt' not in self.word2ind.keys()
        
        assert len(self.ind2word) == Vocab.VOCAB_SIZE
        if trimEmbedding:  
            newEmbedding = dict()
            
            for w in self.ind2word:  
                newEmbedding[w] = self.wordEmbedding[w] 
            self.wordEmbedding=newEmbedding  
            self.saveEmbedding()

    def get_filteredword(self,w):
        if w in self.word2ind.keys():
            return w
        return self.specialWords['EXTRA']
    
    def fit_caption_tokens(self,tokens,length,addPrefix,addSuffix):
        tok = []
        tokens = tokens[0:length]  # get length
        if addPrefix:
            tok.append(self.specialWords['START']) # add prefix
        tok.extend(tokens)  
        if addSuffix: 
            tok.append(self.specialWords['END']) 
        for i in range(length-len(tokens)): 
            tok.append(self.specialWords['NONE'])
        return tok
    
    def onehot_word(self,w):
        encode = [0] * Vocab.VOCAB_SIZE  
        encode[self.word2ind[w]] = 1 
        return encode
    
    def word_fromonehot(self, onehot):
        index = np.argmax(onehot) # find 1
        return self.ind2word[index]
    
    def get_caption_encoded(self,caption,glove, addPrefix, addSuffix):
        tokens = caption_tokenize(caption) 
        tokens = self.fit_caption_tokens(tokens, Vocab.CAPTION_LEN, addPrefix, addSuffix)
        tokens = [self.get_filteredword(x) for x in tokens]
       
        if glove: 
            return [self.wordEmbedding[x] for x in tokens] # use word embedding
        else:
            return [self.onehot_word(x) for x in tokens] # use One-Hot Encoding
        
    def get_caption_from_indexs(self,indx):
        s = ' '.join([self.ind2word[x] for x in indx]) 
        return s
    
    def vocabBuilder():
        vHandler = VideoHandler(VideoHandler.s_fname_train, VideoHandler.s_fname_test)
        train_ids = vHandler.get_otrain_ids() # get video id
        captionData = vHandler.getCaptionData() # get topic
        vocab = Vocab(captionData, train_ids) 
        return [vHandler, vocab]
    
    if __name__ == "__main__":
        vocabBuilder()