import streamlit as st
import pandas as pd

st.title('🤖 Machine Learning App')

st.info('This app builds a machile learning module')

with st.expander('Data'):
  st.write('**Raw Data**')
  df =pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
  df
 
  st.write('**X**')
  X =df.drop('species',axis=1)
  X

  st.write('**Y**')
  Y =df.species
  Y

with st.expander('Data Visualization'):
  st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g',color='species')

#Data preparation
with st.sidebar:
  st.header('Input Features')
  island=st.selectbox('Island',{'Biscoe','Dream','Torgersen'})
  gender = st.selectbox('Gender',('male','female'))
  bill_length_mm=st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
  bill_depth_mm= st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
  flipper_length_mm= st.slider('Flipper Length (mm)', 172.0, 231.0, 201.0)
  body_mass_g= st.slider('Body mass(g)',2700.0,6300.0,4207.0)
