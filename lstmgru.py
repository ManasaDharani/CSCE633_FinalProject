import numpy
import pandas 
import math
from sklearn.model_selection import train_test_split
from keras.models import Input
from keras.models import Model
from keras.layers import LSTM
from sklearn.metrics import f1_score
from keras.layers import Bidirectional, SpatialDropout1D, Conv1D
from keras.layers import CuDNNGRU,CuDNNLSTM
from keras.layers import concatenate,GlobalAveragePooling1D,GlobalMaxPooling1D
from keras.layers import Dense
#Create the embedding dictionary
embedding={}
for text in open('../input/embeddings/glove.840B.300d/glove.840B.300d.txt'):
    embedding[text.split(" ")[0]] = numpy.asarray(text.split(" ")[1:], dtype='float32')             
#read training data from the csv file
train=pandas.read_csv("../input/train.csv")
#read testing data from the csvfile
test_data=pandas.read_csv("../input/test.csv")
#split the training data to get some validation data
split=numpy.random.rand(len(train))<0.85
train_data=train[split]
validation_data=train[~split]
#the lengths of train, test adn validation data
len_train_data=len(train_data)
len_validation_data=len(validation_data)
len_test_data=len(test_data)
print(len_train_data)
print(len_validation_data)
print(len_test_data)
print(train_data.shape)
print(validation_data.shape)
print(test_data.shape)
#Method to create word vectors for the questions
def custom_vector(text):
    temp=[embedding.get(x,numpy.zeros(300)) for x in text[:-1].split()[:100]]
    temp=temp+[numpy.zeros(300)]*(100-len(temp))
    return numpy.array(temp)
#Method to generate batches for test and validation data
def generate_batches(data,length):
    for i in range(length//256):
        batch = test_data.iloc[256*i:256*(i+1),1]
        yield numpy.array([custom_vector(text) for text in batch])
#Method to generate batches for training data
def generate_batches_train(data,length):
    while True:
        for i in range(length//256):
            batch = data.iloc[256*i:256*(i+1),1]
            text_arr = numpy.array([custom_vector(text) for text in batch])
            yield text_arr, numpy.array(data["target"][256*i:256*(i+1)])
#Building the model
to_input=Input(shape=((100,300)))
sd = SpatialDropout1D(0.25)(to_input)
#Add biderectional layers of lstm and gru
gru=Bidirectional(CuDNNGRU(96,return_sequences=True))(sd)
gru=Conv1D(128,kernel_size=2,kernel_initializer="he_uniform")(gru)
lstm=Bidirectional(CuDNNLSTM(96,return_sequences=True))(sd)
lstm=Conv1D(128,kernel_size=2,kernel_initializer="he_uniform")(lstm)
#concatenate lstm and gru
final=concatenate([GlobalAveragePooling1D()(gru),GlobalMaxPooling1D()(gru),GlobalAveragePooling1D()(lstm),GlobalMaxPooling1D()(lstm)])
model=Model(inputs=to_input,outputs=Dense(1,activation="sigmoid")(final))
#Compile the model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()
#Create vectors for validation questions
validation_vectors=numpy.array([custom_vector(text) for text in validation_data["question_text"][0:5000]])
validation_target=numpy.array(validation_data["target"][0:5000])
#Fit the model
model.fit_generator(generate_batches_train(train_data,len_train_data),steps_per_epoch=1000,epochs=50,validation_data=(validation_vectors, validation_target))
#predict using model on validation data
test_validation=[]
for ip in generate_batches(validation_data,len_validation_data):
    test_validation=test_validation+list(model.predict(ip).flatten())
#predict using the model on testing data
test_predicted=[]
for ip in generate_batches(test_data,len_test_data):
    test_predicted=test_predicted+list(model.predict(ip).flatten())
validation_target = validation_data["target"].values
#Calculate the F1 score 
final_score=0
for i in numpy.arange(0.1,0.99,0.01):
    score=f1_score(validation_target,(numpy.array(test_validation)>i).astype(numpy.int))
    if(final_score<score):
        final_score=score
print(final_score)