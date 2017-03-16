import re
import json
from keras.preprocessing import sequence
from keras.models import model_from_json

def clean(s,z=""):
    "clean sentences"
    words = s.split(' ')
    PUNC = '_,.:?!"()*-;/\\][{}]&-' + "'"
    URL = ["http",'ftp','://','.com','.net','.edu','.org','.uk','.gov']
    #stemmer = PorterStemmer()
    
    for i in range(len(words)): 
        words[i] = words[i].strip(PUNC)
        #words[i] = stemmer.stem(words[i])
        if words[i].find('@') != -1:
            for it in URL[3:]:
                if words[i].find(it) != -1:
                    words[i] = '<email>'
        
        for it in URL:
            if words[i].find(it) != -1:
                words[i] = '<url>'
    ret = re.sub('[' + re.escape(PUNC) + ']',' ',' '.join(words)).strip()   
    ret = re.sub('[^'+chr(0)+'-'+chr(177)+']','',ret).strip()
    ret = ret.lower()
    ret = re.sub(r'[\n\r]( )*',' ',ret).strip()
    ret = re.sub(r'( )*[\n\r]',' ',ret).strip()
    ret = re.sub(r'[/&]',' ',ret).strip()
    ret = re.sub(r', ',' ',ret).strip()
    ret = re.sub(r',','',ret).strip()
    ret = re.sub(r'( )+',' ',ret).strip()
    ret = re.sub(r'^(18|19|20)[0-9][0-9][^0-9]','<y> ',ret)
    ret = re.sub(r'^(18|19|20)[0-9][0-9]$','<y>',ret)
    ret = re.sub(r'[^0-9](18|19|20)[0-9][0-9][^0-9]',' <y> ',ret)
    ret = re.sub(r'[^0-9](18|19|20)[0-9][0-9]$',' <y>',ret)
    months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
    for it in months: ret = re.sub(it,'<m>',ret)
    ret = re.sub('<m> to <m>','<m> - <m>',ret)
    ret = re.sub('<y> to <m>','<y> - <m>',ret)
    ret = re.sub('<m> to <y>','<m> - <y>',ret)
    ret = re.sub('<y> to <y>','<y> - <y>',ret)
    if z == 'skills':
        ret = re.sub('[0-9]+(\+)? year(s?)', '', ret)
    return ret.strip()

def getLabel(x):
    labels = {'edu': 1, 'job': 0, 'skills': 2, 'awards': 3, 'link':4, 'certification': 5, 'national_service': 6,'patent': 7, 'group': 8, 'additional': 9}
    labels = {it[1]:it[0] for it in labels.items()}   
    import numpy as np
    return labels[np.argmax(x)]
    
def getLabels(x):
    return [getLabel(it) for it in x]

model = model_from_json(open("models/model.json").read())
#model.load_weights("models/best.model.h5")
model.load_weights("models/best.model.theano.h5")

'''
from keras import backend as K
from keras.utils.np_utils import convert_kernel
for layer in model.layers[2].layers:
	if layer.__class__.__name__ in ['Convolution1D','Convolution2D']:
		original_w = K.get_value(layer.W)
		converted_w = convert_kernel(original_w)
		K.set_value(layer.W,converted_w)

model.save_weights('models/best.model.theano.h5')

exit(0)
'''

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
for instance in sentences:
	sentence = clean(instance[:])
	sentence = [it for it in re.split('[^a-z<>]',sentence) if it]
	for i,word in enumerate(sentence):
		if word.strip() in word2idx:
			sentence[i] = word2idx[word.strip()]
		else:
			sentence[i] = 0
	#sentence = [it for it in sentence if it != -1]
	trainingInstances.append(sentence)

print("Transforming instances")

trainingInstances = sequence.pad_sequences(trainingInstances,maxlen=3104,value=0.)

print("Predicting")
Y_pred = model.predict(trainingInstances,batch_size=1,verbose=1)
print(Y_pred)
print(getLabels(Y_pred))
