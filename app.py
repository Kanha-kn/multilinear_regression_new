import joblib
import numpy as np
import streamlit as st
st.title("Sales predictor")
st.image("naresh_it.jpeg")

TV=st.number_input("please enter sales of Tv")
Newspaper=st.number_input("please enter sales of Newspaper")
Radio=st.number_input("please enter sales of Radio")

model=joblib.load("model.pkl")

if st.button("prediction"):
    features=np.array([[TV,Newspaper,Radio]])
    output=model.predict(features)
    st.write(f"the total sales is {output[0]}") 