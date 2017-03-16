from keras.preprocessing import sequence
from keras.models import model_from_json

def getLabel(x):
    labels = {'edu': 1, 'job': 0, 'skills': 2, 'awards': 3, 'link':4, 'certification': 5, 'national_service': 6,'patent': 7, 'group': 8, 'additional': 9}
    labels = {it[1]:it[0] for it in labels.items()}   
    import numpy as np
    return labels[np.argmax(x)]
    
def getLabels(x):
    return [getLabel(it) for it in x]

model = model_from_json(open("models/model.json").read())
model.load_weights("models/best.model.h5")

def load_vocab(vocab_path):
	data = json.loads(open(vocab_path,'r').read())
	word2idx = data
	idx2word = dict([(v,k) for k,v in data.items()])
	return word2idx, idx2word

word2idx,idx2word = load_vocab("vocab.voc")
print(idx2word[0])

trainingInstances = []

sentences = open("test.txt","r").readlines()
for i in range(len(sentences)): sentences[i] = sentences[i].strip()

print("Reading training")
for instance in trainSentences:
	sentence = instance[:]
	sentence = [it for it in re.split('[^a-z<>]',sentence) if it]
	for i,word in enumerate(sentence):
		if word.strip() in word2idx:
			sentence[i] = word2idx[word.strip()]
		else:
			sentence[i] = 0
	#sentence = [it for it in sentence if it != -1]
	trainingInstances.append(sentence)

print("Transforming instances")

trainingInstances = sequence.pad_sequences(trainingInstances,maxlen=maxLenSentence,value=0.)

print("Predicting")
Y_pred = model.predict(trainingInstances,batch_size=1,verbose=1)
print(Y_pred)
