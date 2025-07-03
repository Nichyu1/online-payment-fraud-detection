# 🛡️ Online Payment Fraud Detection

A Streamlit web app for detecting online payment fraud using machine learning.  
This project allows users to predict fraudulent transactions individually or in bulk, visualize analytics, and explore the model’s performance.

---

## 🚀 Features

- **Single Transaction Prediction:**  
  Enter transaction details and get an instant fraud prediction with model confidence.

- **Batch Prediction:**  
  Upload a CSV file of transactions and receive fraud predictions for each, with downloadable results.

- **Analytics Dashboard:**  
  Visualize fraud rates and transaction type distributions from your batch data.

- **Modern UI:**  
  Clean, user-friendly interface with icons, images, and explainer video.

---

## 🖥️ Demo

> ![image](https://github.com/user-attachments/assets/abcea25b-b1c4-40a6-8373-d9630955fd68)


---

## 📦 Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/Nichyu1/online-payment-fraud-detection.git
   cd online-payment-fraud-detection
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Run the app:**
   ```sh
   streamlit run app.py
   ```

---

## 📂 Dataset

> **Note:**  
> The full dataset (`onlinefraud.csv`) is not included in this repository due to size limits.  
> You can download it from [your Google Drive/Kaggle/other link here] and place it in the project folder.

---

## 📝 Usage

- **Single Prediction:**  
  Fill in the transaction details in the web form and click "Predict".

- **Batch Prediction:**  
  Upload a CSV file with columns:  
  `step,type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest,isFlaggedFraud`  
  Click "Predict on Uploaded CSV" to get results and analytics.

---

## 📊 Example Input

```yaml
Step (Time): 1
Transaction Type: TRANSFER
Amount: 181.00
Old Balance (Origin): 181.00
New Balance (Origin): 0.00
Old Balance (Destination): 0.00
New Balance (Destination): 0.00
Is Flagged Fraud?: 0
```

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/)
- [scikit-learn](https://scikit-learn.org/)
- [Pandas](https://pandas.pydata.org/)
- [Your dataset source, e.g., Kaggle](#)
