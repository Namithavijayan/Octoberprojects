import sys
import numpy
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import sequential
from keras.layers import Dense,Dropout,LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
file=open('Downloads\\frankenstein.txt').read()
#Standardisation
def tokensize_words(input):
    input=input.lower
    tokenizer=RegexpTokenizer(r'\w+')
    tokens=tokenizer.tokenize(input)
    filtered=filter(lambda token: token not in stopwords.words('english'),tokens)
    return "".join(filtered)
processed_input=word_tokenize(file)    
#chars to number
chars=sorted(list(set(processed_input)))
char_to_num=dict((c,i) for i,c in enumerate(chars))
#Checking word to char and char to num has worked
input_len=len(processed_input)
vocab_len=len(chars)
print("Total number of characters",input_len)
print("Total vocab",vocab_len)
seq_length=100
x_data=[]
y_data=[]
for i in range (0,input_len-seq_length,1):
    in_seq=processed_input[i:i+seq_length]
    out_seq=processed_input[i+seq_length]
    x_data.append([char_to_num[char] for char in in_seq])
    y_data.append(char_to_num[out_seq])
    n_pattern=len(x_data)
print("Total Patterns:",n_pattern)
x=numpy.reshape(x_data,(n_pattern,seq_length,1))
x=x/float(vocab_len)
#one-hot encoding
y=np_utils.to_categorical(y_data)
#creating the model
model=sequential()
model.add(LSTM(256,input_shape=[x.shape[1],x.shape[2]],return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256,return_sequence=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1],activation='softmax'))
#compiling the model
model.compile(loss='Categorical_crossentropy',optimizer='adam')
#Saving weights
filepath="model_weights_saved.hdf5"
checkpoint=ModelCheckpoint(filepath,monitor='loss',verbose=1,save_best_only=True,mode='min')
desired_callbacks=[checkpoint]
#Fit model and train
model.fit(x,y,epoches=4,batch_size=256,callbacks=deserved_callbacks)
#Recompile model with saved weights
filename="model_weights_saved.hdf5"
model.load_weights(filename)
model.compile(loss='Categorical_crossentropy',optimizer='adam')
#Output of model back into characters
num_to_char=dict((i,c) for i,c in enumerate(chars))
#Random seed
start=numpy.random.randit(0,len(x_data)-1)
pattern=x_data[start]
print("Random Seed:")
print("\"",''.join([num_to_char[value] for value in pattern]"\""))
#Generating text
for i in range(1000):
    x=numpy.reshape(pattern,(1,len(pattern),1))
    x=x/float(vocab_len)
    pred=model.predict(x,verbose=0)
    index=numpy.argmax(pred)
    result=num_to_char[index]
    seq_in=[num_to_char[value for value in pattern]]
    sys.stdout.write(result)
    pattern.append(index)
    pattern=pattern[1:len(pattern)]

