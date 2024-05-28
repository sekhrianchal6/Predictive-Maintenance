from flask import Flask, request, render_template_string
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import io
import matplotlib.pyplot as plt

app = Flask(__name__)
model = pickle.load(open('lstm.pkl', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))
pca = pickle.load(open('pca.sav', 'rb'))
data = pd.read_csv(r'C:/Users/ADMIN/Documents/pyNotebooks/PyFiles/Modified_Raw.csv')
# Transform data for LSTM Model
inp_data = data.dropna()


def preprocess(data):
    # Exponential Weighted Mean on Test Data
    data = data.drop(['Sensor1', 'Sensor2', 'Sensor3', 'Sensor4', 'Sensor8','Sensor9', 'Sensor13', 'Sensor19', 'Sensor21', 'Sensor22'], axis=1)
    ewm_data = data.ewm(10).mean()
    scaled_data = scaler.transform(ewm_data)
    pca_data = pca.transform(scaled_data)
    data = data.drop(['Sensor1', 'Sensor2', 'Sensor3', 'Sensor4', 'Sensor5', 'Sensor6', 'Sensor8','Sensor9', 'Sensor13', 'Sensor19', 'Sensor21', 'Sensor22', 'Sensor24'], axis=1)
    ewm_data = data.ewm(10).mean()
    scaled_data = scaler.transform(ewm_data)
    pca_data = pca.transform(scaled_data)
    scaled_df = pd.DataFrame(columns = ewm_data.columns, data = scaled_data)
    scaled_df['PC1'] = pca_data[:, 0]
    return scaled_df
    
@app.route('/')
def index():
    return '''
        <h1>File Content Display</h1>
        <form method="post" action="/handle_buttons">
            <label for="filepath">Enter file path:</label>
            <input type="text" id="filepath" name="filepath">
            <button type="submit" name="action" value="load">Load File</button>
            <button type="submit" name="action" value="predict">Predict</button>
        </form>
    '''

@app.route('/handle_buttons', methods=['POST'])
def handle_buttons():
    filepath = request.form['filepath']
    action = request.form['action']
    if action == 'load':
        if os.path.exists(filepath):
            with open(filepath, 'r') as file:
                content = file.read()
            return render_template_string('''
                <h1>File Content Display</h1>
                <form method="post" action="/handle_buttons">
                    <label for="filepath">Enter file path:</label>
                    <input type="text" id="filepath" name="filepath">
                    <button type="submit" name="action" value="load">Load File</button>
                    <button type="submit" name="action" value="predict">Predict</button>
                </form>
                <h2>Content of {{ filepath }}</h2>
                <pre>{{ content }}</pre>
                <form action="/">
                    <button type="submit">Load Another File</button>
                </form>
            ''', filepath=filepath, content=content)
        else:
            return render_template_string('''
                <h1>File Content Display</h1>
                <form method="post" action="/handle_buttons">
                    <label for="filepath">Enter file path:</label>
                    <input type="text" id="filepath" name="filepath">
                    <button type="submit" name="action" value="load">Load File</button>
                    <button type="submit" name="action" value="predict">Predict</button>
                </form>
                <p style="color: red;">File not found: {{ filepath }}</p>
                <form action="/">
                    <button type="submit">Try Again</button>
                </form>
            ''', filepath=filepath)
    elif action == 'predict':
        # Placeholder for predict action
        data = preprocess(inp_data)
        test_data = np.array(data.iloc[-30:, :]) # Consider last 30 data only
        test_data = np.expand_dims(test_data, axis=0)
        pred = model.predict(test_data)
        print(pred)
        plot_line()
        return render_template_string('''
            <h1>Prediction Feature</h1>
            <p>This is where the prediction logic will go. {{ pred }}</p>
            <form action="/">
                <button type="submit">Go Back</button>
            </form>
        ''')

def plot_line():
    a = np.ones(10)
    plt.figure()
    df.plot(a)
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title('Line Plot')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
