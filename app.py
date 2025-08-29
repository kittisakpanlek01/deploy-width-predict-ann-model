import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from tensorflow import keras

# Load the trained model and scaler
model = keras.models.load_model('width_prediction_model.keras')
scaler = joblib.load('scaler.pkl')
scaler_y = joblib.load('scaler_y.pkl')
st.title("Width Prediction App")
# Choose input method
input_mode = st.radio("เลือกวิธีการกรอกข้อมูล", ["กรอกข้อมูลเอง", "อัพโหลดไฟล์ Excel"])
if input_mode == "อัพโหลดไฟล์ Excel":
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])
    if uploaded_file is not None:
        input_df = pd.read_excel(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(input_df)
        # required_cols = ['TEMTAR','ActWidthIn','RMEXTW','PSSPOS_More','PESPOSIn','PESPOSOut','INDH']
        required_cols = ['RMEXTW','TEMTAR','ActWidthIn','PSSPOS_More','PESPOSIn','PESPOSOut','INDH']
        # st.write("Columns in scaler:", scaler.feature_names_in_)
        # st.write("Columns in uploaded file:", input_df.columns.tolist())

        # ตรวจว่ามีครบไหม
        if not set(required_cols).issubset(input_df.columns):
            st.error(f"Excel file must contain columns: {list(required_cols)}")
        else:
            input_df = input_df[list(required_cols)]  # จัดเรียงให้ตรงกับ scaler
            input_data_scaled = scaler.transform(input_df)
            predictions_scaled = model.predict(input_data_scaled)
            predictions = scaler_y.inverse_transform(predictions_scaled)
            input_df['Predicted Width'] = predictions
            st.dataframe(input_df)
    else:
        st.write("กรุณาอัพโหลดไฟล์ Excel เพื่อทำการทำนาย")

    # uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])
    # if uploaded_file is not None:
    #     input_df = pd.read_excel(uploaded_file)
    #     st.write("Uploaded Data:")
    #     st.dataframe(input_df)
    #     required_cols = ['TEMTAR','ActWidthIn','RMEXTW','PSSPOS_More','PESPOSIn','PESPOSOut','INDH']

    #     # ตรวจว่ามีครบไหม
    #     if not set(required_cols).issubset(input_df.columns):
    #         st.error(f"ไฟล์ Excel ต้องมี columns: {required_cols}")
    #     else:
    #         input_df = input_df[required_cols]  # จัดเรียง column ให้ตรงกับ scaler
    #         input_data_scaled = scaler.transform(input_df)
    #         predictions_scaled = model.predict(input_data_scaled)
    #         predictions = scaler_y.inverse_transform(predictions_scaled)
    #         input_df['Predicted Width'] = predictions
    #         st.write("Predictions:")
    #         st.dataframe(input_df)

    # else:
    #     st.write("กรุณาอัพโหลดไฟล์ Excel เพื่อทำการทำนาย")
else:
    st.write("กรุณากรอกข้อมูลด้านล่าง:")
    # Input fields for features
    temtar = st.number_input("TEMTAR", value=1100.0)
    actwidthin = st.number_input("ActWidthIn", value=1219.0)
    rmextw = st.number_input("RMEXTW", value=1229.0)
    psspos_more = st.number_input("PSSPOS_More", value=80.0)
    pesposin = st.number_input("PESPOSIn", value=1230.0)
    pesposout = st.number_input("PESPOSOut", value=1250.0)
    indh = st.number_input("INDH", value=3.0)
    # Predict button
    if st.button("Predict Width"):
        input_data = np.array([[temtar, actwidthin, rmextw, psspos_more, pesposin, pesposout, indh]])
        input_data_scaled = scaler.transform(input_data)
        prediction_scaled = model.predict(input_data_scaled)
        prediction = scaler_y.inverse_transform(prediction_scaled)
        st.success(f"Predicted Width: {prediction[0][0]:.2f}")
    else:
        st.write("กรุณากรอกข้อมูลเพื่อทำการทำนาย")
