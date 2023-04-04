import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn

csvfile = "X.csv"
transformfile = "finalized_transform.sav"
modelfile = "finalized_model.sav"


st.title('Bandung House Price Predict')


@st.cache_data
def openalldata(csvfile, transformfile, modelfile):
    file = pd.read_csv(csvfile)
    transform = pickle.load(open(transformfile, 'rb'))
    mlfile = pickle.load(open(modelfile, 'rb'))

    return file, transform, mlfile

fileloc, powertrans, model = openalldata(csvfile, transformfile, modelfile)


def predict_price(location, bedroom, toilet, carport, sqm_lot, sqm_living):
    loc_index = np.where(fileloc.columns == location)[0][0]

    x = np.zeros(len(fileloc.columns))
    x[0] = bedroom
    x[1] = toilet
    x[2] = carport
    x[3] = sqm_lot
    x[4] = sqm_living
    if loc_index >= 0:
        x[loc_index] = 1
    x = powertrans.transform(np.array(x).reshape(1, len(x)))
    predict = model.predict(x)[0]
    return predict


regions = []

for i in fileloc.columns:
    if "Bandung" in i:
        regions.append(i)

region = st.selectbox('Choose the region', regions)
bedroom = st.number_input('Bedroom', 0)
toilet = st.number_input('Toilet', 0)
carport = st.number_input('Carport', 0)
sqm_lot = st.number_input('Lot Area (m²)')
sqm_living = st.number_input('Building Area (m²)')
button = st.button('Make prediction')

if button:
    pricepred = predict_price(region, bedroom, toilet, carport, sqm_lot, sqm_living)
    st.success("Your house price is Rp{:,.2f}".format(pricepred))

