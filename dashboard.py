

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# Load Data
st.title("Incident Reporting Dashboard")
uploaded_file = st.file_uploader("Upload your firewall log file (CSV)", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.fillna(0, inplace=True)
    
    # Feature Engineering
    reference_point = pd.Timestamp("2024-01-01 00:00:00")  # Adjust reference
    df['Timestamp'] = reference_point + pd.to_timedelta(df['Elapsed Time (sec)'], unit='s')
    df['hour'] = pd.to_datetime(df['Timestamp']).dt.hour
    df['Avg Packet Size Sent'] = df['Bytes Sent'] / (df['pkts_sent'] + 1e-6)
    df['Avg Packet Size Received'] = df['Bytes Received'] / (df['pkts_received'] + 1e-6)
    df['Traffic Rate'] = df['Bytes'] / (df['Elapsed Time (sec)'] + 1e-6)
    df['Packet Rate'] = df['Packets'] / (df['Elapsed Time (sec)'] + 1e-6)
    action_mapping = {'drop': 0, 'deny': 1, 'allow': 2, 'reset-both': 3}
    df['action_encoded'] = df['Action'].map(action_mapping)
    df['Bytes Imbalance'] = df['Bytes Sent'] - df['Bytes Received']
    df['Bytes Imbalance (%)'] = (abs(df['Bytes Sent'] - df['Bytes Received']) / (df['Bytes'] + 1e-6)) * 100
    df['Packet Ratio'] = df['pkts_sent'] / (df['pkts_received'] + 1e-6)
    df['Log Packet Ratio'] = np.log1p(df['Packet Ratio'])
    
    # Anomaly Detection
    features = ['Source Port', 'Destination Port', 'Traffic Rate', 'Packet Rate', 'Log Packet Ratio']
    X = df[features]
    model = IsolationForest(contamination=0.05, random_state=42)
    df['anomaly'] = model.fit_predict(X)
    anomalies = df[df['anomaly'] == -1]
    
    # Dashboard Sections
    # 1. Incident Summaries
    st.header("Incident Summaries")
    time_frame = st.selectbox("Select Timeframe", ["Hourly", "12 Hours", "24 Hours"])
    
    if time_frame == "Hourly":
        summary = df.groupby('hour')[['Bytes', 'Packets']].sum()
    elif time_frame == "12 Hours":
        df['12_hour'] = pd.cut(df['hour'], bins=[0, 12, 24], labels=['0-12', '12-24'], right=False)
        summary = df.groupby('12_hour')[['Bytes', 'Packets']].sum()
    else:  # 24 Hours
        summary = df[['Bytes', 'Packets']].sum().to_frame(name="Total").T
    
    st.write(summary)
    
    # 2. Error Insights
    st.header("Error Insights")
    errors = df[df['Action'].isin(['drop', 'deny'])]
    st.write(f"Total Errors: {len(errors)}")
    st.dataframe(errors)
    
    # 3. Threat Detection
    st.header("Threat Detection")
    st.write(f"Total Threats Detected: {len(anomalies)}")
    st.dataframe(anomalies)
    
    # 4. Traffic Volume Analysis
    st.header("Traffic Volume Analysis")
    hourly_traffic = df.groupby('hour')['Bytes'].sum()
    st.line_chart(hourly_traffic, use_container_width=True)
    
    # 5. Rare IP Activity
    st.header("Rare IP Activity")
    rare_sources = df['Source Port'].value_counts().nsmallest(5)
    st.write("Rare Source Ports:")
    st.write(rare_sources)
    
    rare_destinations = df['Destination Port'].value_counts().nsmallest(5)
    st.write("Rare Destination Ports:")
    st.write(rare_destinations)
    
    # 6. Unexpected Traffic Spikes
    st.header("Unexpected Traffic Spikes")
    traffic_mean = df['Traffic Rate'].mean()
    traffic_std = df['Traffic Rate'].std()
    spikes = df[df['Traffic Rate'] > (traffic_mean + 3 * traffic_std)]
    st.write(f"Total Traffic Spikes Detected: {len(spikes)}")
    st.dataframe(spikes)
    
else:
    st.warning("Please upload a valid CSV file to start the analysis.")
