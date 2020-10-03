from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import numpy as np
def scaling(weather):
    weather = scaler.fit_transform(weather)
    return weather
def windowing(weather,cut=595,window=17):
    test = weather[cut:(-window)]
    ds = tf.data.Dataset.from_tensor_slices(test)
    ds = ds.window(window + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window + 1))
    ds = ds.shuffle(100)
    ds = ds.map(lambda w: (w[:-1], tf.squeeze(w[-1:])))
    return ds.batch(window).prefetch(1)
def gettemp(testbatches):
    path=r"weather.h5"
    model = keras.models.load_model(path)
    x,y = next(iter(testbatches))
    output = model.predict(x)
    p = output.reshape(-3,3)
    p = scaler.inverse_transform(p)
    p=p.reshape(-1,1)
    forecast_rain = []
    forecast_min=[]
    forecast_max = []
    for i in range(0,len(p),51):
        forecast_max.append(p[i])
        forecast_min.append(p[i+1])
        forecast_rain.append(p[i+2])
    return forecast_max,forecast_min,forecast_rain

def wantedvalues(x):
    path=r"weather_data.npy"
    weather=np.load(path)
    weather=scaling(weather)
    weather = windowing(weather)
    tmax,tmin,rain=gettemp(weather)
    x=int(x)
    tempformonth = "<h1>Data for the month{}</h1> <br/> Maximum Temperature : {} <br/> Minimum Temperature :{} <br/> Rain : {}".format(x,tmax[x],tmin[x],rain[x])
    return tempformonth

from flask import Flask,render_template, request
app = Flask(__name__)

@app.route("/",methods=["GET","POST"])
def getyourvalue():
    content = request.json
    month = content["month"]
    reqd= wantedvalues(month)
    return reqd

if (__name__=='__main__'):
    app.run(debug=True)