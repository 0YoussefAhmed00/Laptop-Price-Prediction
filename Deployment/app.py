import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

coreencoder = pickle.load(open('core_encoder.pkl', 'rb'))
decision = pickle.load(open('Decision.pkl', 'rb'))
gpytypeencoder = pickle.load(open('GPU_Type_encoder.pkl', 'rb'))
modelencoder = pickle.load(open('model_encoder.pkl', 'rb'))
ramtypeencoder = pickle.load(open('Ram_type_encoder.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('f.html')


@app.route('/predict', methods=['POST'])
def predict():
    # For rendering results on HTML GUI

    # Retrieve form inputs
    spec = float(request.form['spec'])
    ram = float(request.form['ram'])
    ram_type = request.form['ramtype']
    rom = float(request.form['rom'])
    display_size = float(request.form['displaysize'])
    width = int(request.form['width'])
    height = int(request.form['height'])
    model = request.form['model']  # Assuming 'model' is a string input
    core = request.form['core']
    thread = int(request.form['thread'])
    gpu_type = request.form['gputype']

    ##applying the encoding part to the columns: model, gpu type, core and ram type
    ram_type_encoded = ramtypeencoder.get(ram_type)
    model_encoded = modelencoder.get(model)
    gpu_encoded = gpytypeencoder.get(gpu_type)
    core_encoded = coreencoder.get(core)

    # Combine features into a single array
    features = np.array([[spec, ram, ram_type_encoded, rom, display_size, width, height, model_encoded,
                          core_encoded, thread, gpu_encoded]])

    # Apply normalization to the features
    final_features = scaler.transform(features)

    # Make a prediction
    prediction = decision.predict(final_features)

    # Render the result to HTML
    return render_template('f.html', prediction_text=prediction)


if __name__ == "__main__":
    app.run(debug=True)