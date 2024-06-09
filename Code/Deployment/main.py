from flask import Flask, request, render_template_string, send_file, render_template
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import io
import matplotlib.pyplot as plt
import plotly.express as px
from dotenv import load_dotenv
from plotly import utils
from json import dumps

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
eda_data=pd.read_csv('train_FD001.csv')
pd.DataFrame.iteritems = pd.DataFrame.items

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

def get_shape(data):
    rows=data.shape[0]
    col=data.shape[1]
    
    
    return rows,col
    
@app.route('/eda')
def eda():
    rows,col=get_shape(eda_data)
    df_head=eda_data.head()
    df_describe=eda_data.describe().T
    figure=parallel_coord(eda_data)
    
    return render_template('eda.html',rows=rows,col=col,tables=[df_head.to_html(classes="table table-stripped")],titles=df_head.columns.values,desc_tables=[df_describe.to_html(classes="table table-stripped")],desc_titles=df_describe.columns.values,figure=figure)

@app.route('/team')
def team():
    return render_template('team.html')

def parallel_coord(data):
    print(type(data))
    fig = px.parallel_coordinates(data.iloc[:,2:], color="Remaining Cycles",
                                color_continuous_scale=px.colors.diverging.delta_r,
                                color_continuous_midpoint=2,width=1200,height=1200,labels={'Sensor1':'S1', 'Sensor2':'S2', 'Sensor3':'S3', 'Sensor4':'S4',
        'Sensor5':'S5', 'Sensor6':'S6', 'Sensor7':'S7', 'Sensor8':'S8', 'Sensor9':'S9', 'Sensor10':'S10',
        'Sensor11':'S11', 'Sensor12':'S12', 'Sensor13':'S13', 'Sensor14':'S14', 'Sensor15':'S15', 'Sensor16':'S16',
        'Sensor17':'S17', 'Sensor18':'S18', 'Sensor19':'S19', 'Sensor20':'S20', 'Sensor21':'S21', 'Sensor22':'S22',
        'Sensor23':'S23', 'Sensor24':'S24', 'Remaining Cycles':'RC'})
    json_fig=dumps(fig,cls=utils.PlotlyJSONEncoder)

    return json_fig

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


