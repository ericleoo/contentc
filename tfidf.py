folder = "C:\\Users\\I337902\\content\\"
fi = open(folder + "E.csv","r").readlines()
fi = [it[:-1].split(',') for it in fi]
Y  = [it[0] for it in fi]
X = [it[1] for it in fi]

from sklearn.feature_extraction.text import TfidfVectorizer
#vect = TfidfVectorizer(stop_words = 'english',ngram_range = (1,2),min_df = 5)
vect = pickle.load(open(folder + "vect_final_2.pkl","rb"))

X = vect.transform(X)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight = 'balanced', solver = 'sag', multi_class = 'multinomial', max_iter = 1000000)
model.fit(X,Y)

fi = open(folder + "F.csv","r").readlines()
fi = [it[:-1].split(',') for it in fi]
Y_test  = [it[0] for it in fi]
X_test = [it[1] for it in fi]
X_test = vect.transform(X_test)
Y_pred = model.predict(X_test)

from sklearn.metrics import *
f1_score(Y_test,Y_pred,average = 'macro')
f1_score(Y_test,Y_pred,average = 'micro')
print(classification_report(Y_test,Y_pred))

import pickle
pickle.dump(vect,open(folder + "vect_bal_multi.pkl","wb"))
pickle.dump(model,open(folder + "model_bal_multi.pkl","wb"))
