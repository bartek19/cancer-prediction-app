import streamlit as st
import pickle
import pandas as pd
from columns import column_labels
import plotly.graph_objects as go
import numpy as np


def get_clean_data():
    data = pd.read_csv("../data/data.csv")
    
    data = data.drop(["Unnamed: 32", "id"], axis=1)
    
    data["diagnosis"] = data["diagnosis"].map({ "M": 1, "B": 0})
    
    return data

def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    
    data = get_clean_data()
    
    input_dict = {
        
    }
    
    for key, label in column_labels.items():
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(data[key].min()),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    
    return input_dict

def get_scaled_values(input_dict):
    data = get_clean_data()
    
    X = data.drop(['diagnosis'], axis=1)
    
    scaled_dict = {}
    
    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
        
    return scaled_dict

def get_radar_chart(input_data):
    
    input_data = get_scaled_values(input_data)
    
    categories = ['Texture', 'Symmetry', 'Fractal Dimension', 
                  'Compactness', 'Concavity', 'Concave Points', 'Radius', 
                  'Perimeter', 'Area', 'Smoothness']

    
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[input_data["texture_mean"], input_data["symmetry_mean"], input_data["fractal_dimension_mean"], 
           input_data["compactness_mean"], input_data["concavity_mean"], input_data["concave points_mean"], 
           input_data["radius_mean"], input_data["perimeter_mean"], input_data["area_mean"], input_data["smoothness_mean"]
           ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[input_data["texture_se"], input_data["symmetry_se"], input_data["fractal_dimension_se"], 
           input_data["compactness_se"], input_data["concavity_se"], input_data["concave points_se"], 
           input_data["radius_se"], input_data["perimeter_se"], input_data["area_se"], input_data["smoothness_se"]
            ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[input_data["texture_worst"], input_data["symmetry_worst"], input_data["fractal_dimension_worst"], 
           input_data["compactness_worst"], input_data["concavity_worst"], input_data["concave points_worst"], 
           input_data["radius_worst"], input_data["perimeter_worst"], input_data["area_worst"], input_data["smoothness_worst"]
            ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
    showlegend=True
    )

    return fig

def add_predictions(input_data):
    model = pickle.load(open("/Users/bartoszmeller/STREAMLIT-APP-CANCER/model/model.pkl", "rb"))
    scaler = pickle.load(open("/Users/bartoszmeller/STREAMLIT-APP-CANCER/model/scaler.pkl", "rb"))
    
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    
    input_array_scaled = scaler.transform(input_array)
    
    prediction = model.predict(input_array_scaled)
    
    st.header("Cell cluster prediction")
    st.write("The cell cluster is:")
    
    if prediction[0] == 0:
        st.write("Benign")
    else: 
        st.write("Malicious")
    
    st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being malicious: ", model.predict_proba(input_array_scaled)[0][1])
    st.write("This is app is not intended to replace professional medical diagnosis. Do not use as the only source of decision-making proccess.")


def main():
    st.set_page_config(
        page_title = "Breast Cancer Predictor",
        page_icon = ":feamle-doctor:",
        layout = "wide",
        initial_sidebar_state= "expanded"
    )
    
    input_data = add_sidebar()
    
    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("Tweak the sliders on the side and explore different conditions to check the probability of breast cancer. This app predicts (using ML model) whether you have anything to worry about. Please note that it is not 100% accurate and does not replace the medical opinion.")
    
    col1, col2 = st.columns([4,1])
    
    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)
          

if __name__ == "__main__":
    main()