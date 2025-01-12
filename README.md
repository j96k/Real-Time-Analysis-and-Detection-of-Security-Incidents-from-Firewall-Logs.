# Problem Statement: Real-Time Analysis and Detection of Security Incidents from Firewall Logs

## Approach

### Overview
The goal of this project is to analyze network traffic data and detect anomalies, patterns, and performance issues using various metrics. The dataset contains traffic information such as source and destination ports, packet rates, traffic rates, and packet ratios. Through multiple steps, the data is transformed, analyzed, and classified to identify potential issues and derive insights.

### Steps

1. **Elapsed Time to Timestamps**  
   The project begins by converting elapsed time (in seconds) into timestamps based on a fixed reference point. This allows us to track when network events occurred and analyze traffic patterns over time.

2. **Error Insights**  
   We filter the dataset to identify rows where actions are labeled as 'drop' or 'deny'. These actions are treated as errors, and the number of such errors is counted. This step helps in monitoring network issues and failed transactions.

3. **Traffic Patterns**  
   The project calculates the average packet size for both sent and received traffic. These metrics help understand the efficiency of data transmission and identify potential bottlenecks or inefficiencies in the network.

4. **Rate Calculations**  
   The traffic rate (bytes per second) and packet rate (packets per second) are calculated to understand the data transfer speeds. These values give insights into the overall traffic load and can help in detecting any unusual spikes in traffic.

5. **Packet Ratio**  
   The packet ratio (packets sent to packets received) is calculated to assess the balance of incoming and outgoing traffic. A higher or lower packet ratio can indicate network issues such as congestion, inefficiencies, or security concerns.

6. **Logarithmic Transformation of Packet Ratio**  
   A logarithmic transformation is applied to the packet ratio to normalize the data. This transformation helps in reducing the impact of outliers and makes the data more suitable for analysis, especially when there are large variations in packet ratio values.

7. **Rare IP Activity Detection**  
   Rare source and destination ports are identified by counting the occurrences of each port and filtering for the least common ones. This helps identify unusual or suspicious traffic patterns, which may indicate potential security threats or misconfigured systems.

8. **Anomaly Detection with Isolation Forest**  
   Using the Isolation Forest algorithm, anomalies in network traffic are detected based on selected features such as source and destination ports, traffic rate, and packet rate. The algorithm isolates anomalies by identifying data points that deviate significantly from the rest of the dataset. Anomalies are marked, and their count is displayed to assess the overall network health.

9. **Classifying Log Packet Ratio**  
   The Log Packet Ratio is classified into three categories: 'Acceptable', 'Warning', and 'Critical'. This classification helps in quickly assessing the severity of network traffic and identifying areas that may require attention, such as critical traffic that could indicate performance issues or potential security risks.

### Conclusion
This approach offers a comprehensive framework for analyzing network traffic, detecting anomalies, and identifying performance issues. By calculating various metrics such as packet ratios, traffic rates, and identifying rare IP activities, we can gain valuable insights into the network's health. The use of anomaly detection models like Isolation Forest further enhances the ability to detect unusual patterns, while classification techniques allow for efficient prioritization of network events.


## ML Approach 1: Machine Learning for Anomaly Detection

The goal of this approach is to classify network traffic data as either normal or anomalous by identifying **Critical** and **Warning** traffic patterns. This allows for the simplification of anomaly categories and facilitates easy interpretation of results, enabling effective detection and alerting when anomalies occur.

### Approach Overview

1. **Labeling Anomalies**  
   The dataset's **Log Packet Ratio Label** is used to create a new label, **Anomaly_Label**, which combines **Critical** and **Warning** traffic into one category, marked as '1'. All other traffic is marked as '0' to indicate normal behavior. This binary labeling simplifies the anomaly detection process.

2. **Feature Selection**  
   The features used for training the model include network traffic metrics such as bytes sent and received, packets sent and received, elapsed time, and action-encoded labels. These features are essential for detecting abnormal patterns in network traffic.

3. **Model Training and Testing**  
   A **Random Forest Classifier** is used to train the model on the labeled dataset. The model is evaluated on a held-out test set, with metrics such as accuracy, precision, recall, and F1-score computed to assess the performance.

4. **Evaluation Metrics**  
   - **Accuracy:** The model achieved an accuracy of 100% on the test data.
   - **Precision, Recall, and F1-Score:** The classification report shows perfect precision, recall, and F1-scores for both normal and anomalous traffic, indicating that the model effectively detects both normal behavior and anomalies.

### Results
- **Accuracy:** 100%
- **Precision:** 1.00 (for both normal and anomalous traffic)
- **Recall:** 1.00 (for both normal and anomalous traffic)
- **F1-Score:** 1.00 (for both normal and anomalous traffic)

### Conclusion
This machine learning approach successfully classifies network traffic into normal and anomalous categories. The model demonstrates excellent performance with 100% accuracy, and the results can be used to alert when network behavior deviates from the expected patterns. By simplifying the categories into a binary classification (normal vs. anomalous), the model's predictions are easy to interpret and act upon, making it a powerful tool for anomaly detection in network traffic analysis.

## ML Approach 2: Machine Learning with Class Imbalance Handling

The second approach aims to identify normal behavior and anomalies (critical/warning) in network traffic data, similar to the first approach, but with additional handling for class imbalance. This is achieved by computing class weights to give more importance to the less frequent class (anomalous traffic), improving the model's performance in detecting anomalies.

### Approach Overview

1. **Labeling Anomalies**  
   As in the first approach, the dataset's **Log Packet Ratio Label** is used to create a binary label called **Anomaly_Label**. Traffic labeled as **Critical** or **Warning** is considered anomalous and marked as '1', while all other traffic is considered normal and marked as '0'.

2. **Class Weight Calculation**  
   To address potential class imbalance, class weights are computed using a balanced strategy. This ensures that the model does not bias the majority class (normal traffic) and gives appropriate attention to detecting anomalies, which may be underrepresented in the dataset.

3. **Model Training and Testing**  
   A **Random Forest Classifier** is used to train the model with the computed class weights. The model is evaluated on a held-out test set, and the performance metrics, including accuracy, precision, recall, and F1-score, are computed.

4. **Evaluation Metrics**  
   - **Accuracy:** The model achieved an accuracy of 100% on the test data.
   - **Precision, Recall, and F1-Score:** The classification report shows perfect precision, recall, and F1-scores for both normal and anomalous traffic, indicating that the model effectively detects both normal behavior and anomalies, even when class imbalance is present.

### Results
- **Accuracy:** 100%
- **Precision:** 1.00 (for both normal and anomalous traffic)
- **Recall:** 1.00 (for both normal and anomalous traffic)
- **F1-Score:** 1.00 (for both normal and anomalous traffic)

### Conclusion
This approach improves upon the first by incorporating class weight adjustments to handle imbalanced datasets. Despite the potential imbalance between normal and anomalous traffic, the model achieved perfect accuracy and classification metrics, ensuring reliable anomaly detection in network traffic. The class weight adjustment makes the model more robust, enhancing its ability to detect rare anomalous events.

## ML Approach 3: Multi-Class Classification with Class Weight Handling

The third approach aims to classify network traffic data into three categories: **Acceptable**, **Critical**, and **Warning**. The model incorporates class weight adjustments to handle the imbalance in the distribution of traffic labels. The **Warning** category is highly underrepresented, and handling this imbalance helps improve the model's ability to classify anomalies.

### Approach Overview

1. **Label Encoding**  
   The **Log Packet Ratio Label** is encoded into numerical labels to allow for multi-class classification. This process converts the categorical labels (Acceptable, Critical, Warning) into numeric values for the machine learning model.

2. **Class Weight Calculation**  
   Class weights are computed to address the class imbalance. Since the **Warning** category is extremely underrepresented, it receives a higher weight, ensuring the model pays more attention to detecting this rare class.

3. **Model Training and Testing**  
   A **Random Forest Classifier** is trained with the computed class weights, and the model is evaluated on a held-out test set. Predictions are made on the test set, and performance metrics including accuracy, precision, recall, and F1-score are computed for each class.

### Results
- **Accuracy:** 100%
- **Precision:** 1.00 (for both Acceptable and Critical traffic), 1.00 for Warning (but with a recall of 0.33)
- **Recall:** 1.00 (for Acceptable and Critical traffic), 0.33 for Warning
- **F1-Score:** 1.00 (for Acceptable and Critical traffic), 0.50 for Warning

### Conclusion
In this approach, the model achieved perfect accuracy and classification for most traffic categories. However, due to the extremely low number of **Warning** samples, the recall for this category is significantly lower. The class weight handling helped to account for the class imbalance, improving the detection of rare classes, but the low number of **Warning** instances made it challenging to predict this category effectively. Despite this, the model showed robust performance in detecting **Acceptable** and **Critical** traffic.


## Prerequisites
1. A **Google Colab** account or **VSCode** installed on your local machine.
2. Download the dataset from [Kaggle - Internet Firewall Data Set](https://www.kaggle.com/datasets/tunguz/internet-firewall-data-set).
3. Place the dataset in the `data/log2.csv` folder.

## How to Run
1. Open `examine_firewall_logs.ipynb` in Google Colab.
2. Install dependencies by running the first code cell.
3. Upload the dataset to Google Drive and update the path in the code. 
4. Run all cells sequentially.

## Dependencies
The following libraries are used in the notebook. Install them in Google Colab before running the code:
'bash
pip install -r requirements.txt
pip install pandas
pip install sklearn
pip install matplotlib

## Reporting Dashboard for Firewall Logs

This dashboard provides a summary of security incidents from firewall logs. It displays:
- Incident summaries for different timeframes: hourly, 12 hours, and 24 hours.
- Error metrics such as network failures and packet loss.
- Threats like intrusion attempts and malware communication.
- Unusual activities such as unexpected traffic spikes and rare IP activities.

The dashboard is built using **Streamlit** for interactivity and **pandas** for data processing.

## Prerequisites
1. **Python 3.7+** installed on your system.
2. Install the required libraries using the following command:
   ```bash
   pip install -r requirements.txt
   pip install streamlit pandas numpy scikit-learn

## How to Run
1. Place the dataset in the `data/output.csv` folder.
2. Run the script: `streamlit run dashboard.py`
