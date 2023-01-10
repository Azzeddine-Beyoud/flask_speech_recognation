from flask import Flask,request,jsonify
import numpy as np
import pickle
from keras.models import load_model
from keras.models import model_from_json
import os
from werkzeug.utils import secure_filename
import librosa
import IPython.display as ipd
import numpy as np
from keras.models import load_model
# from gevent.pywsgi import WSGIServer

# Import Keras dependencies
from tensorflow.python.framework import ops
ops.reset_default_graph()
from keras.preprocessing import image

# Import other dependecies
import numpy as np
import h5py
from PIL import Image
import PIL

# model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

# model = load_model('/home/azzeddine/Desktop/souad_tp/models/best_model.hdf5')

# Model files
MODEL_ARCHITECTURE = '/home/azzeddine/Desktop/souad_tp/models/new_model_weight/model_adam.json'
MODEL_WEIGHTS = '/home/azzeddine/Desktop/souad_tp/models/new_model_weight/model_weights.h5'

# Load the model from external files
json_file = open(MODEL_ARCHITECTURE)
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)


# Get weights into the model
model.load_weights(MODEL_WEIGHTS)

# print('Model loaded. Check http://127.0.0.1:5000/')

train_audio_path = '/home/azzeddine/Desktop/souad_tp/dataset/train/train/audio'
classes = ['bed','bird','cat','dog', 'down','eight','five','four', 'go', 'happy', 'house', 'left', 'marvin','nine','no',
 'off', 'on','one','right','seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'wow', 'yes', 'zero']

def predict(audio):
    prob=model.predict(audio.reshape(1,8000))
    index=np.argmax(prob[0])
    return classes[index]

# ::: MODEL FUNCTIONS :::
def model_predict(img_path, model):

    # Pre-processing
    samples, sample_rate = librosa.load(img_path, sr = 16000)
    samples = librosa.resample(samples, sample_rate, 8000)
    # ipd.Audio(samples, rate= 8000)
	
    # model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    prediction = predict(samples)

    return prediction

# ::: FLASK ROUTES
@app.route('/')
def index():
    # basepath = os.path.dirname(__file__)
    # print(basepath)
    return "Hello world"


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    # train_audio_path = '/home/azzeddine/Desktop/souad_tp/dataset/train/train/audio'
    # classes = os.listdir(train_audio_path)

    if request.method == 'POST':

		# Get the file from post request
        f = request.files['file']

		# Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        print(basepath)
        file_path = os.path.join(
			basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

		# Make a prediction
        # prediction = model_predict('/home/azzeddine/Desktop/souad_tp/test/yes.wav', model)
        prediction = model_predict(file_path, model)

        # predicted_class = classes[prediction[0]]
        # print('We think that is {}.'.format(predicted_class.lower()))

        # return str(predicted_class).lower()
        return f'pred {prediction}'

# @app.route('/predict',methods=['POST'])
# def predict():
#     cgpa = request.form.get('cgpa')
#     iq = request.form.get('iq')
#     profile_score = request.form.get('profile_score')
#     input_query = np.array([[cgpa,iq,profile_score]])

#     # result = {'cgpa': cgpa, 'iq' : iq ,'profile_score' : profile_score}
#     # return jsonify(result)

#     # result = model.predict(input_query)[0]
#     result = model.predict('/home/azzeddine/Desktop/souad_tp/test/six.wav')[0]
#     return jsonify({'placement':str(result)})

if __name__ == '__main__':
    app.run(debug=True)


