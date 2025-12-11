# Customer Churn Prediction with Artificial Neural Networks

A machine learning project that predicts customer churn in banking using a trained Artificial Neural Network (ANN). This project includes model training, evaluation, and an interactive web application built with Streamlit.

## 🎯 Project Overview

This project aims to predict whether a customer will leave a bank based on their demographic and financial information. The model is trained using TensorFlow/Keras on customer data and deployed as an interactive web application for real-time predictions.

**Live Demo:** [Customer Churn Predictor](https://cutomerchurn.streamlit.app/)

## 📊 Dataset

- **Source:** `assets/annclassification/Churn_Modelling.csv`
- **Features:** 11 input features including:
  - Credit Score
  - Geography (France, Germany, Spain)
  - Gender
  - Age
  - Tenure (years with bank)
  - Balance
  - Number of Products
  - Has Credit Card (binary)
  - Is Active Member (binary)
  - Estimated Salary
- **Target:** Churn (0 = stays, 1 = leaves)

## 🏗️ Project Structure

```
deep-learning-projects/
├── app.py                          # Streamlit web application
├── Exp1.ipynb                      # Data exploration & model training notebook
├── prediction.ipynb                # Prediction examples notebook
├── requirements.txt                # Python dependencies
├── ann_model.h5                    # Trained ANN model (saved format)
├── scaler.pkl                      # StandardScaler for feature normalization
├── label_encoder_gender.pkl        # LabelEncoder for gender feature
├── OneHotGeo.pkl                   # OneHotEncoder for geography feature
├── assets/
│   └── annclassification/
│       └── Churn_Modelling.csv     # Dataset file
├── logs/                           # Training logs
└── README.md                       # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ (tested with Python 3.13)
- pip or conda

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/LabsVelns/deep-learning-projects.git
   cd deep-learning-projects
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Run the Web Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` and allow you to:
- Input customer details
- Get real-time churn predictions
- View prediction confidence scores

### Run Model Training & Exploration

Open the Jupyter notebooks:
```bash
jupyter notebook Exp1.ipynb      # Training & exploration
jupyter notebook prediction.ipynb # Prediction examples
```

## 🧠 Model Architecture

The ANN model consists of:
- **Input Layer:** 11 features
- **Hidden Layer 1:** 128 neurons + ReLU activation
- **Hidden Layer 2:** 64 neurons + ReLU activation
- **Hidden Layer 3:** 32 neurons + ReLU activation
- **Output Layer:** 1 neuron + Sigmoid activation (binary classification)

**Compilation Details:**
- **Optimizer:** Adam
- **Loss Function:** Binary Crossentropy
- **Metrics:** Accuracy, AUC

## 📦 Dependencies

All dependencies are listed in `requirements.txt`:
- **numpy** — Numerical computing
- **pandas** — Data manipulation
- **scikit-learn** — Machine learning utilities (encoding, scaling, metrics)
- **tensorflow** — Deep learning framework
- **keras** — High-level neural networks API
- **matplotlib** — Visualization
- **seaborn** — Statistical visualization
- **streamlit** — Web app framework

Install all at once:
```bash
pip install -r requirements.txt
```

## 📈 Model Performance

The model achieves strong predictive performance on the test set:
- **Accuracy:** ~86%
- **Precision & Recall:** Balanced across both classes
- See `Exp1.ipynb` for detailed evaluation metrics and confusion matrices

## 🔧 Data Preprocessing

1. **Feature Scaling:** StandardScaler normalizes numerical features
2. **Categorical Encoding:**
   - Gender: LabelEncoder (binary: 0 or 1)
   - Geography: OneHotEncoder (3 countries → 3 binary columns)
3. **Train-Test Split:** 80% train, 20% test

## 💾 Model Files

The trained model and encoders are serialized and included in the repo:
- `ann_model.h5` — Trained Keras model (HDF5 format)
- `scaler.pkl` — StandardScaler for inverse transformations
- `label_encoder_gender.pkl` — Gender label encoder
- `OneHotGeo.pkl` — Geography one-hot encoder

## 🖥️ Web Application Features

The Streamlit app (`app.py`) provides:
- **Interactive Input Form:** Sliders and dropdowns for all customer features
- **Real-Time Predictions:** Instant churn probability scores
- **Model Output:** Displays prediction and confidence level
- **User-Friendly Interface:** Clean, intuitive design

## 📝 Usage Example

```python
# Load model and scalers
from keras.models import load_model
import pickle
import pandas as pd

model = load_model('ann_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare input
input_data = pd.DataFrame({
    'CreditScore': [750],
    'Gender': [1],  # Encoded
    'Age': [35],
    'Tenure': [5],
    'Balance': [50000],
    'NumOfProducts': [2],
    'HasCrCard': [1],
    'IsActiveMember': [1],
    'EstimatedSalary': [100000]
})

# Scale and predict
scaled_data = scaler.transform(input_data)
prediction = model.predict(scaled_data)
churn_probability = prediction[0][0]
print(f"Churn Probability: {churn_probability:.2%}")
```

## 🧪 Testing & Validation

Detailed model evaluation and testing are available in:
- `Exp1.ipynb` — Model training, evaluation, and visualization
- `prediction.ipynb` — Prediction examples and manual testing

## 🐛 Troubleshooting

### Model File Not Found
- Ensure `ann_model.h5` is in the project root directory
- For Streamlit Cloud deployment, commit the model file to GitHub

### Pickle Load Errors
- Ensure all pickle files (`.pkl`) are in the project root
- Python version compatibility: pickle files created with Python 3.13 may need to be regenerated on different versions

### Missing Dependencies
- Run `pip install -r requirements.txt` to install all required packages

## 🚀 Deployment

### Local Deployment
```bash
streamlit run app.py
```

### Streamlit Cloud Deployment
1. Push the project to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repo
4. Deploy by selecting this repo and `app.py` as the main file

**Note:** Ensure `requirements.txt` and all model files (`.h5`, `.pkl`) are committed to the repository.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**LabsVelns**

- GitHub: [@LabsVelns](https://github.com/LabsVelns)
- Project: [Deep Learning Projects](https://github.com/LabsVelns/Deep-Learning-Projects)

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## 📧 Contact & Support

For questions or issues:
- Open an [Issue](https://github.com/LabsVelns/Deep-Learning-Projects/issues) on GitHub
- Contact via GitHub profile

---

**Last Updated:** December 2025

Happy Predicting! 🎯

