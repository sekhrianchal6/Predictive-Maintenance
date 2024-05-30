from flask import Flask, request, render_template_string, send_file, render_template
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import io
import matplotlib.pyplot as plt

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
    return '''
        <h1 style="width: 100%;text-align: center;">Predictive Maintenace of NASA Aircraft Engine</h1>
        <h2>Turbo Engine</h2>
        <img src="{{ url_for('static', filename='turbofan.png') }}" alt="Your Image" style="width: 300px; height: auto;">

        <p>The degradation that can occur to aircraft parts over the course of their lifetime directly impacts both their performance and reliability. In order to provide the necessary 
maintenance behavior, this machine learning research will be focused on providing a framework for predicting the aircraft's remaining usable life (RUL) based on the whole life 
cycle data. Using the NASA C-MAPSS data set is implemented and tested to determine the engine lifetime.

Tracking and predicting the progression of damage in aircraft engine turbo machinery has some roots in the work of Kurosaki. They estimate the efficiency and the flow rate deviation
of the compressor and the turbine based on operational data, and utilize this information for fault detection purposes.
 
A low pressure compressor (LPC) and high pressure compressor (HPC) supply compressed high temperature, high pressure gases to the combustor. Low pressure turbine (LPT) can 
decelerate and pressurize air to improve the chemical energy conversion efficiency of aviation kerosene. High pressure turbines (HPT) generate mechanical energy by using high 
temperature and high pressure gas strike turbine blades. Low-pressure rotor (N1), high-pressure rotor (N2), and nozzle guarantee the combustion efficiency of the engine.

Our main focus will be on accurately recording low RUL values to prevent putting the engine at danger and forecasting the RUL of the turbofan engine while accounting for HPC 
failure. Data analysis, data visualization and Model development are covered in this project</p>
        
        <form method="post" action="/handle_buttons">
            <button type="submit" name="action" value="predict">Predict</button>
        </form>
    '''

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

if __name__ == '__main__':
    app.run(debug=True)
