from __future__ import absolute_import
import numpy as np
import os
import json
import itertools
import codecs
import sys
from collections import Counter
import re
import argparse
from sklearn.metrics import classification_report
np.random.seed(0)

# Parameters
# ==================================================
#
# Model Variations. See Kim Yoon's Convolutional Neural Networks for 
# Sentence Classification, Section 3 for detail.

# output model directory
# need a different model directory per configuration
model_dir = 'models/' # 'models10' refer to the 10th model in my experiment

# Training parameters
embeddingsDim = 200
batchSize = 128
hiddenSize = 200
posSize = 100
epochs = 1000
maxLenSentence = 0

parser = argparse.ArgumentParser()
parser.add_argument('-train',action='store',dest='trainFileName',help='Training File Path')
parser.add_argument('-embeddings', action='store', dest='embeddingsDim', help='Embeddings dimension',default=200)
parser.add_argument('-output',action='store',dest='outputName',help='Output file name (including models)')
parser.add_argument('-batch',action='store',dest='batchSize', help='batch size',default=32)
parser.add_argument('-vocab',action='store',dest='vocab',help='vocab size',default=10000)
parser.add_argument('-hidden',action='store',dest='hiddenSize',help='hidden layer size',default=200)
parser.add_argument('-maxSequenceLength',action='store',dest='maxSequenceLength',help='hidden layer size',default=700)
parser.add_argument('-verbose',action='store',dest='verbose',help='verbose',default=1)
parser.add_argument('-test',action='store',dest='testFileName',help='Testing File Path')
parser.add_argument('-tune',action='store',dest='tuneFileName',help='Tuning File Path')

parameters = parser.parse_args()

embeddingsDim = int(parameters.embeddingsDim)
batchSize = int(parameters.batchSize)
hiddenSize = int(parameters.hiddenSize)
outputName = parameters.outputName
trainFileName = parameters.trainFileName
vocabSize = int(parameters.vocab)
verb = int(parameters.verbose)
testFileName = parameters.testFileName
tuneFileName = parameters.tuneFileName
maxSequenceLength = int(parameters.maxSequenceLength)
outFile = codecs.open(outputName + ".out",'w')

print("embeddingsDim: " + str(embeddingsDim))
print("batchSize: " + str(batchSize))
print("hiddenSize: " + str(hiddenSize))
print("outputName: " + outputName)
print("trainFileName: " + trainFileName)
print("tuneFileName: " + tuneFileName)
print("testFileName: " + testFileName)
print("Vocab size: " + str(vocabSize))
print("maxSequenceLength: " + str(maxSequenceLength))


outFile.write("embeddingsDim: " + str(embeddingsDim) + "\n")
outFile.write("batchSize: " + str(batchSize) + "\n")
outFile.write("hiddenSize: " + str(hiddenSize) + "\n")
outFile.write("outputName: " + outputName + "\n")
outFile.write("trainFileName: " + trainFileName + "\n")
outFile.write("tuneFileName: " + tuneFileName + "\n")
outFile.write("testFileName: " + testFileName + "\n")
outFile.write("Vocab size: " + str(vocabSize) + "\n")
outFile.write("maxSequenceLength: " + str(maxSequenceLength) + "\n")
outFile.write("\n-------------------------------------------------\n\n")

from keras.preprocessing import sequence
from keras.models import Model, Sequential, model_from_json
from keras.layers import *
from keras.layers.wrappers import TimeDistributed
from keras.utils.np_utils import to_categorical

def load_vocab(vocab_path):
	data = json.loads(open(vocab_path,'r').read())
	word2idx = data
	idx2word = dict([(v,k) for k,v in data.items()])
	return word2idx, idx2word

def word2vec_embedding_layer(embeddings_path,y):
	weights = np.load(open(embeddings_path, 'rb'))
	layer = Embedding(input_dim = weights.shape[0], input_length = y, output_dim = weights.shape[1],weights = [weights],trainable = False)
	return layer
	
word2idx, idx2word = load_vocab("vocab.voc")

temp = {'edu': 1, 'job': 0, 'skills': 2, 'awards': 3, 'link':4, 'certification': 5, 'national_service': 6,'patent': 7, 'group': 8, 'additional': 9}
labels = [0 for i in range(10)]
for it in temp.items(): labels[it[1]] = it[0]

trainingInstances = []
trainingLabels = []
tuningInstances = []
tuningLabels = []

trainSentences = open("E.csv","r").readlines()
for i in range(len(trainSentences)): trainSentences[i] = trainSentences[i].strip().split(',')
tuneSentences = open("F.csv","r").readlines()
for i in range(len(tuneSentences)): tuneSentences[i] = tuneSentences[i].strip().split(',')

print("Reading training")
for instance in trainSentences:
	sentence = ' <break> '.join(instance[1:])
	sentence = [it for it in re.split('[^a-z<>]',sentence) if it]
	labs = instance[0]
	for i,word in enumerate(sentence):
		if word.strip() in word2idx:
			sentence[i] = word2idx[word.strip()]
		else:
			sentence[i] = 0
	#sentence = [it for it in sentence if it != -1]
	trainingInstances.append(sentence)
	
	trainingLabels.append(labels.index(labs[:]))
	if len(sentence) > maxLenSentence:
		maxLenSentence = len(sentence)

print("Reading tuning")
for instance in tuneSentences:
	sentence = ' <break> '.join(instance[1:])
	sentence = [it for it in re.split('[^a-z<>]',sentence) if it]
	labs = instance[0]
	for i,word in enumerate(sentence):
		if word.strip() in word2idx:
			sentence[i] = word2idx[word.strip()]
		else:
			sentence[i] = 0
	#sentence = [it for it in sentence if it != -1]
	tuningInstances.append(sentence)
	tuningLabels.append(labels.index(labs[:]))
	
#if maxLenSentence > maxSequenceLength:
#maxLenSentence = maxSequenceLength
	
print("Transforming labels")
	
trainingLabels = to_categorical(trainingLabels,nb_classes=len(labels))
tuningLabels = to_categorical(tuningLabels,nb_classes=len(labels))

print("Saving labels")

labelsFile = open('labels.txt','w')
labelsFile.write("\n".join(labels))
labelsFile.close()

print("Transforming instances")

trainingInstances = sequence.pad_sequences(trainingInstances,maxlen=maxLenSentence,value=0.)
tuningInstances = sequence.pad_sequences(tuningInstances,maxlen=maxLenSentence,value=0.)


# Building model
# ==================================================
#
# graph subnet with one input and one output,
# convolutional layers concateneted in parallel
print("Building model")

fsz = 3
dropout_prob = (0.2,0.2,0.2,0.1)
num_filters = 20

graph_in = Input(shape=(maxLenSentence, 20))
conv = Convolution1D(nb_filter=num_filters,
					 filter_length=fsz,
					 border_mode='same',
					 activation='relu',
					 subsample_length=1)(graph_in)
#pool = MaxPooling1D(pool_length=2)(conv)
#pool = BatchNormalization()(pool)
'''
pool = Merge(mode = 'concat', concat_axis = 1)([conv,graph_in])
conv = Convolution1D(nb_filter=num_filters,
					 filter_length=fsz,
					 border_mode='same',
					 activation='relu',
					 subsample_length=1)(pool)
#pool = MaxPooling1D(pool_length=2)(conv)
#pool = BatchNormalization()(pool)
'''
#pool = Merge(mode = 'concat', concat_axis = 1)([conv,graph_in])
out = Flatten()(conv)

graph = Model(input=graph_in, output=out)

# main sequential model

main_input = Input(shape = (maxLenSentence,), dtype = 'int32', name = 'main_input')
#x = word2vec_embedding_layer("embedding.emb",maxLenSentence)(main_input)
x = Embedding(80448,20,input_length = maxLenSentence)(main_input)
y = graph(x)
x = Dense(len(labels))(y)
out1 = Activation('softmax')(x)
model = Model(input = main_input, output = out1)
#model.load_weights(model_dir+"best.model.h5")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['fmeasure'])


model_json = model.to_json()
fname = model_dir + 'model.json'
with open(fname, "w") as json_file:
    json_file.write(model_json)

bestScore = 0
for e in range(epochs):
	print("Epoch: " + str(e))
	outFile.write("Epoch: " + str(e) + "\n")
	model.fit(trainingInstances,trainingLabels,batch_size=batchSize,nb_epoch=1,verbose=verb,class_weight='auto')

	print("Predicting tuning...")
	
	predictions = model.predict(tuningInstances,batch_size=batchSize,verbose=verb)
	outLabels = []
	
	for i,row in enumerate(predictions):
		row = row.tolist()
		labs = []
		for j,score in enumerate(row):
			if float(score) >= 0.5:
				labs.append(tuple([score,labels[j]]))
		labs.sort(reverse=True)
		labs = ",".join([it[1] for it in labs])
		outLabels.append(str(i) + "\t" + labs)
	
	fo = codecs.open('temp/output.' + outputName + '.tune.' + str(e) +'.txt','w')
	fo.write("\n".join(outLabels))
	fo.close()

	scores = model.evaluate(tuningInstances,tuningLabels)
	
	outFile.write("Tuning...\n")
	for i in range(len(model.metrics_names)):
		outFile.write(model.metrics_names[i] + "\t" + str(scores[i]*100) + '\n')
		print("%s: %.2f%%" % (model.metrics_names[i], scores[i]*100))
		
	Y_true = []
	Y_pred = []	
		
	for i in range(len(tuningInstances)):
		softmax = predictions[i]
		temp = sorted([(val,idx) for idx,val in enumerate(softmax)], reverse = True)
		fnd = False
		
		y_true = 0
		for j in range(len(tuningLabels[i])):
			if tuningLabels[i][j] > 0.5:
				y_true = j
				break
		Y_true.append(y_true)
		
		for j in range(1):
			prob = temp[j][0]
			y_pred =  temp[j][1]
			if y_pred == y_true:
				fnd = True
				Y_pred.append(y_pred)
				break
		
		if not fnd: Y_pred.append(temp[0][1])
	
	outFile.write(classification_report(Y_true,Y_pred) + '\n')
	print(classification_report(Y_true,Y_pred))
	
	if bestScore <= scores[-1]:
		bestScore = scores[-1]
		fname = model_dir + 'best.model.h5'
		model.save_weights(fname,overwrite=True)

outFile.close()

