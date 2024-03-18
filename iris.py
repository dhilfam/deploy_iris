import streamlit as st
import pandas as pd
import pickle
import time
from PIL import Image

st.set_page_config(page_title="Halaman Modelling", layout="wide")

st.write("""
# Welcome to my Machine Learning Dashboard

This ML dashboard created by: [M.Fadhil](https://www.linkedin.com/in/muhammad-fadhil-18aba6259/)
""")

st.write("""
# Iris Species Prediction App

This website application is crafted to forecast Iris Species by leveraging data from the iris dataset provided by UCIML.
Whether you possess a profound interest in botany or are an enthusiast of data science, this predictive tool serves as your portal to delve into the realm of flower classification and the possibilities of machine learning.
Let's embark on a journey to unveil the beauty and intricacies concealed within the petals of these magnificent iris flowers.
""")

add_selectitem = st.sidebar.selectbox("Want to open about?", (" ", "Iris species!"))

def iris():
    st.write("""
    This app predicts the **Iris Species**
    
    Data obtained from the [iris dataset](https://www.kaggle.com/uciml/iris) by UCIML. 
    """)

st.sidebar.header('User Input Features:')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
else:
        def user_input_features():
            st.sidebar.header('Input Manual')
            SepalLengthCm = st.sidebar.slider('Sepal Length (cm)', 4.3,6.5,10.0)
            SepalWidthCm = st.sidebar.slider('Sepal Width (cm)', 2.0,3.3,5.0)
            PetalLengthCm = st.sidebar.slider('Petal Length (cm)', 1.0,4.5,9.0)
            PetalWidthCm = st.sidebar.slider('Petal Width (cm)', 0.1,1.4,5.0)
            data = {'SepalLengthCm': SepalLengthCm,
                    'SepalWidthCm': SepalWidthCm,
                    'PetalLengthCm': PetalLengthCm,
                    'PetalWidthCm': PetalWidthCm}
            features = pd.DataFrame(data, index=[0])
            return features

input_df = user_input_features()
img = Image.open("iris.jpg")
st.image(img, width=500)

if st.sidebar.button('Predict!'):
        df = input_df
        st.write(df)
        with open("generate_iris.pkl", 'rb') as file:  
            loaded_model = pickle.load(file)
        prediction = loaded_model.predict(df)
        result = ['Iris-setosa' if prediction == 0 else ('Iris-versicolor' if prediction == 1 else 'Iris-virginica')]
        st.subheader('Prediction: ')
        output = str(result[0])
        with st.spinner('Wait for it...'):
            time.sleep(4)
            st.success(f"Prediction of this app is {output}")

if add_selectitem == "Iris species!":
    iris()
