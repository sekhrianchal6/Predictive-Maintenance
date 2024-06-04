from flask import Flask, request, render_template_string, send_file, render_template
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import io
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()


app = Flask(__name__)

model = pickle.load(open('lstm.pkl', 'rb'))
model2 = pickle.load(open('mlp_clipped_model.pkl', 'rb'))
model3 = pickle.load(open('xgb_clipped_model.pkl', 'rb'))
model4 = pickle.load(open('CNN_LSTM.pkl', 'rb'))

scaler = pickle.load(open('scaler.sav', 'rb'))
pca = pickle.load(open('pca.sav', 'rb'))
data = pd.read_csv('Modified_Raw.csv')
# Transform data for LSTM Model
inp_data = data.dropna()


def preprocess(data):
    # Exponential Weighted Mean on Test Data
    data = data.drop(['Sensor1', 'Sensor2', 'Sensor3', 'Sensor4', 'Sensor5', 'Sensor6', 'Sensor8','Sensor9', 'Sensor13', 'Sensor19', 'Sensor21', 'Sensor22', 'Sensor24'], axis=1)
    ewm_data = data.ewm(10).mean()
    scaled_data = scaler.transform(ewm_data)
    pca_data = pca.transform(scaled_data)
    scaled_df = pd.DataFrame(columns = ewm_data.columns, data = scaled_data)
    scaled_df['PC1'] = pca_data[:, 0]
    return scaled_df
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/eda')
def eda():
    return render_template('eda.html')

@app.route('/handle_buttons', methods=['POST'])
def handle_buttons():
    action = request.form['action']
    if action == 'predict':
        # Placeholder for predict action
        #LSTM predict
        
        data = preprocess(inp_data.drop('Cycles', axis=1))
        test_data = np.array(data.iloc[-30:, :]) # Consider last 30 data only
        test_data = np.expand_dims(test_data, axis=0)        
        pred = model.predict(test_data)
        
        #MLP predict
        data = preprocess(inp_data.drop(['Cycles'], axis=1))
        data['Cycles'] = inp_data['Cycles']
        test_data = np.array(data.iloc[-1, :]).reshape(1,-1)
        pred2 = model2.predict(test_data)
 
        #XGB predict
        data = preprocess(inp_data.drop(['Cycles'], axis=1))
        data['Cycles'] = inp_data['Cycles']
        test_data = np.array(data.iloc[-1, :]).reshape(1,-1)
        pred3 = model3.predict(test_data) 
        
        #CNN + LSTM predict
        data = preprocess(inp_data.drop('Cycles', axis=1))        
        test_data = np.array(data.iloc[-30:, :]) # Consider last 30 data only
        test_data = np.expand_dims(test_data, axis=0)        
        pred4 = model4.predict(test_data)
        
        #print(pred)
        labels = ['LSTM', 'MLP', 'XGBoost', 'CNN+LSTM']
        data = [pred[0,0], pred2[0], pred3[0], pred4[0,0]]
        
        return render_template('BarChart.html', labels=labels, data=data)#, labels=labels, data=data)




app.config['DEBUG']=os.environ.get('FLASK_DEBUG')

if __name__ == '__main__':
    app.run(debug=True)


