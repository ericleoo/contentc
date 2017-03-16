from keras import backend as K
from keras.utils.np_utils import convert_kernel
from keras.models import model_from_json

model = model_from_json(open("models/model.json").read())
model.load_weights("models/best.model.h5")

for layer in model.layers[2].layers:
	if layer.__class__.__name__ in ['Convolution1D','Convolution2D']:
		original_w = K.get_value(layer.W)
		converted_w = convert_kernel(original_w)
		K.set_value(layer.W,converted_w)

model.save_weights('models/best.model.theano.h5')
exit(0)
