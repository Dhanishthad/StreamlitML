import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('🤖 Machine Learning App')

st.info('This is app builds a machine learning model!')

with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('https://raw.githubusercontent.com/Dhanishthad/StreamlitML/master/dogs_cleaned.csv')
  df

  st.write('**X**')
  X_raw = df.drop('breed', axis=1)
  X_raw

  st.write('**y**')
  y_raw = df.breed
  y_raw

with st.expander('Data visualization'):
  st.scatter_chart(data=df, x='height_cm', y='body_mass_g', color='breed')

# Input features
with st.sidebar:
  st.header('Input features')
  country = st.selectbox('Country', ('UK', 'Canada', 'Germany'))
  height_cm = st.slider('Height (cm)', 32.1, 59.6, 43.9)
  weight_kg = st.slider('Weight (kg)', 13.1, 21.5, 17.2)
  tail_length_cm = st.slider('Tail length (cm)', 172.0, 231.0, 201.0)
  body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
  sex= st.selectbox('Sex', ('male', 'female'))
  
  # Create a DataFrame for the input features
  data = {'country': country,
          'height_cm': height_cm,
          'weight_kg': weight_kg,
          'tail_length_cm': tail_length_cm,
          'body_mass_g': body_mass_g,
          'sex':sex  }
  input_df = pd.DataFrame(data, index=[0])
  input_dogs = pd.concat([input_df, X_raw], axis=0)

with st.expander('Input features'):
  st.write('**Input dogs**')
  input_df
  st.write('**Combined dogs data**')
  input_dogs


# Data preparation
# Encode X
encode = ['country', 'sex']
df_dogs = pd.get_dummies(input_dogs, prefix=encode)

X = df_dogs[1:]
input_row = df_dogs[:1]

# Encode y
target_mapper = {'Labrador': 0,
                 'Golden Retriever': 1,
                 'Bulldog': 2,
                 'German Shepherd':3
                }
def target_encode(val):
  return target_mapper[val]

y = y_raw.apply(target_encode)

with st.expander('Data preparation'):
  st.write('**Encoded X (input dogs)**')
  input_row
  st.write('**Encoded y**')
  y


# Model training and inference
## Train the ML model
clf = RandomForestClassifier()
clf.fit(X, y)

## Apply model to make predictions
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ['Labrador', 'Golden Retriever', 'Bulldog','German Shepherd']
df_prediction_proba.rename(columns={0: 'Labrador',
                                 1: 'Golden Retriever',
                                 2: 'Bulldog',
                                 3: 'German Shepherd'})

# Display predicted species
st.subheader('Predicted breed')
st.dataframe(df_prediction_proba,
             column_config={
               'Labrador': st.column_config.ProgressColumn(
                 'Labrador',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               'Golden Retriever': st.column_config.ProgressColumn(
                 'Golden Retriever',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
               'Bulldog': st.column_config.ProgressColumn(
                 'Bulldog',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
                'German Shepherd': st.column_config.ProgressColumn(
                 'German Shepherd',
                 format='%f',
                 width='medium',
                 min_value=0,
                 max_value=1
               ),
             }, hide_index=True)


dogs_breed = np.array(['Labrador', 'Golden Retriever', 'Bulldog','German Shepherd'])
st.success(str(dogs_breed[prediction][0]))
