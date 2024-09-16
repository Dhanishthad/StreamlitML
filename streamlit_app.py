import streamlit as st

st.title('🤖 Machine Learning App')

st.info('This app builds a machile learning module')

df =pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv')
df

