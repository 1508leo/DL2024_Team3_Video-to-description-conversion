import numpy as np
import os                   # Miscellaneous operating system interfaces
import sys                  # System-specific parameters and functions
import keras.backend as K
import tensorflow as tf


from keras.applications import ResNet50, VGG16 
from keras.applications.inception_v3 import InceptionV3

from keras.applications.resnet50 import preprocess_input # preprocess image

from keras.layers import Dropout, Merge, Flatten, RepeatVector, Activation
from keras.layers import Embedding, Conv2D, MaxPooling2D, LSTM, GRU, BatchNormalization
from keras.layers import TimeDistributed, Dense, Input, Flatten, GlobalAveragePooling2D, Bidirectional
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import RMSprop # Optimizers
from keras.preprocessing import image
from keras.regularizers import l2 # L2 regulization

from backend.vocab import Vocab 

def sentence_distance(y_true, y_pred):
    return K.sqrt(K.sum(K.square(K.abs(y_true-y_pred)),axis=1,keepdims=True))

class VModel:

		def __init__(self, CAPTION_LEN, VOCAB_SIZE, cutoffonly = False, learning = True):
            self.CAPTION_LEN = CAPTION_LEN
            self.VOCAB_SIZE  = VOCAB_SIZE
            if not cutoffonly:
                self.build_mcnn(self.CAPTION_LEN, self.VOCAB_SIZE, learning = learning)
            self.build_cutoffmodel()

        def  get_model(self):
            return self.model
                
        '''
        Attempt to split pretrained CNN out of model
        To cache a lower dimension vector per frame to file
        # PC : pretrained CNN will be non-trainable now
        '''
        # co == Cutoff Model

        def build_cutoffmodel(self):
            base = ResNet50(include_top = False, weights='imagenet') # 
            # base = InceptionV3(include_top = False, weights='imagenet') 
            self.co_model = base # co_model is ResNet50
                
            self.co_model.summary()
            self.co_model._make_predict_function() 
            self.graph = tf.get_default_graph() 
                
            return self.co_model

        def co_getoutshape(self, assert_model = None):
        # ResNet
            shape = (None,2048)
            ## Inception V3
            # shape = (None, 8*8*2048)
            
            assert len(shape) == 2
            assert shape[0] is None
            return shape

        def preprocess_partialmodel(self, frames):
            frames_in = np.asarray([image.img_to_array(frame) for frame in frames]) # 
            frames_in = preprocess_input(frames_in)
            # Create TensorFlow graph
            with self.graph.as_default():
                frames_out = self.co_model.predict(frames_in) # 
                frames_out = np.array([frame.flatten() for frame in frames_out]) # 
            return frames_out

        def train_mode(self):
            K.set_learning_phase(1)

        def build_mcnn(self, CAPTION_LEN, VOCAB_SIZE, learning = True):
            if learning:
                self.train_mode()
            from backend.videohandler import VideoHandler # video
            # cmodel Processing text data
            cmodel = Sequential() # Sequential
            
            cmodel.add(TimeDistributed(Dense(512,kernel_initializer='random_normal'), input_shape=(CAPTION_LEN+1,Vocab.OUTDIM_EMB )))
            cmodel.add(LSTM(512, return_sequences=True,kernel_initializer='random_normal'))
            cmodel.summary()
        
            input_shape_audio = VideoHandler.AUDIO_FEATURE
            # amodel Processing audio feature data
            amodel = Sequential() # Sequential
            amodel.add(GRU(128,
                        dropout=0.2,
                        recurrent_dropout=0.2,
                        return_sequences=True,
                        input_shape=input_shape_audio))
            amodel.add(BatchNormalization())
            amodel.add(GRU(64,
                        dropout=0.2,
                        recurrent_dropout=0.2,
                        return_sequences=True))
            amodel.add(BatchNormalization()) 
            amodel.add(Flatten()) 
            amodel.add(RepeatVector(CAPTION_LEN + 1)) 
            amodel.summary()

            input_shape_vid = self.co_getoutshape()
            # imodel Processing image feature data
            imodel = Sequential()
            imodel.add(TimeDistributed(Dense(1024,kernel_initializer='random_normal'), input_shape=input_shape_vid))
            imodel.add(TimeDistributed(Dropout(0.20)))
            imodel.add(TimeDistributed(BatchNormalization(axis=-1)))
            imodel.add(Activation('tanh')) 
            imodel.add(Bidirectional(GRU(1024, return_sequences=False, kernel_initializer='random_normal')))
            imodel.add(RepeatVector(CAPTION_LEN + 1))

            imodel.summary()

            model = Sequential()
            model.add(Merge([cmodel,amodel,imodel],mode='concat')) # Combine
            model.add(TimeDistributed(Dropout(0.2)))
            model.add(LSTM(1024,return_sequences=True, kernel_initializer='random_normal',recurrent_regularizer=l2(0.01)))
            model.add(TimeDistributed(Dense(VOCAB_SIZE,kernel_initializer='random_normal')))
            model.add(Activation('softmax')) 
            optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0) 
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy']) # Compile
            model.summary()

            self.model = model
            return model

        def plot_model(self, filename):
            from keras.utils import plot_model
            plot_model(self.model, to_file=filename, show_shapes = True, show_layer_names = False)
            print("Model Plotted in %s"%filename)

    if __name__ == "__main__":
        if sys.argv[1] == "plot_model":
            from vocab import Vocab
            vmodel = VModel(Vocab.CAPTION_LEN, Vocab.VOCAB_SIZE)
            vmodel.plot_model(sys.argv[2])