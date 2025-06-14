# 🚀 End-to-End MLOps: Customer Purchase Propensity API 🛍️

This project demonstrates the complete lifecycle of a machine learning model, from data exploration and feature engineering to building a high-performance, containerized REST API for serving real-time predictions.

A key highlight of this project is the **iterative model improvement process**, where an initial baseline model (V1) with poor performance was systematically diagnosed and enhanced to create a high-performance, business-ready model (V2).

**✨ Live API Documentation (Swagger UI):** [https://your-render-service-url.onrender.com/docs](https://your-render-service-url.onrender.com/docs)  
*(Note: The free tier on Render may spin down the service due to inactivity. The first request might take 30-60 seconds to wake the server up.)*

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [The Machine Learning Workflow](#-the-machine-learning-workflow)
  - [1. Data Processing & Feature Engineering](#1-data-processing--feature-engineering)
  - [2. Model Training & Iterative Improvement](#2-model-training--iterative-improvement)
  - [3. Model Serving via REST API](#3-model-serving-via-rest-api)
  - [4. Containerization & Deployment](#4-containerization--deployment)
- [Results & Performance: V1 vs. V2](#-results--performance-v1-vs-v2)
- [How to Run This Project](#-how-to-run-this-project)
- [Future Improvements](#-future-improvements)

## 🎯 Project Overview

The core business problem is to identify e-commerce users who are most likely to make a purchase in the near future. By identifying these high-propensity users, a business can take targeted actions (e.g., personalized discounts, reminders) to increase conversion rates and revenue.

This project tackles the problem by:
1.  **Engineering predictive features** from raw user behavioral data.
2.  **Training and iterating on a high-performance XGBoost model**.
3.  **Serving the final model** as a live REST API using FastAPI.
4.  **Containerizing the entire application** with Docker for portability and easy deployment to the cloud.

The project uses the [Retail Rocket E-commerce Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset).

## 🛠️ Tech Stack

- **Data Science & Modeling:**
  - `Python 3.11`
  - `Pandas` for data manipulation
  - `Scikit-learn` for data splitting and evaluation
  - `XGBoost` for the gradient boosting model
- **API & Engineering:**
  - `FastAPI` for building the high-performance prediction API
  - `Uvicorn` as the ASGI server
  - `Docker` 🐳 for containerization
- **Development Environment & Deployment:**
  - `Google Colab` for initial exploration and model development
  - `Visual Studio Code` for local development and API assembly
  - `Git & GitHub` for version control
  - `Render` ☁️ for cloud deployment

## 📁 Project Structure
The project is organized into distinct modules for clarity and maintainability.
```
.
├── 📁 app/                  # Contains all files for the FastAPI application
│   ├── main.py             # The main API logic
│   ├── propensity_to_buy_model_v2.pkl # The trained model artifact
│   └── features_v2.json    # List of features the model expects
├── 📁 data/                 # (Ignored by Git) Raw data files like events.csv
├── 📁 scripts/              # Reusable scripts for local data processing and training
│   ├── process_data.py
│   └── train_model.py
├── .gitignore              # Specifies files/folders for Git to ignore
├── Dockerfile              # Recipe to build the application's Docker image
├── requirements.txt        # List of Python dependencies
└── README.md               # You are here!
```

## 🔄 The Machine Learning Workflow

### 1. Data Processing & Feature Engineering
The project began in Google Colab to handle the large (2.7M+ events) dataset. The most critical step was engineering features that capture user intent. Instead of basic counts, **time-windowed features** were created (e.g., `num_views_in_last_7_days`, `add_to_cart_rate_in_last_30_days`) to model the recency and intensity of user engagement.

### 2. Model Training & Iterative Improvement
An **XGBoost Classifier** was chosen for its performance. The development process followed a crucial iterative loop:

-   **Model V1 (Baseline):** The first model was trained on simple, globally aggregated features. The result was a model with an extremely low **Precision of 0.03%**, making it commercially unviable as it produced too many false positives.

-   **Model V2 (High-Performance):** After diagnosing V1's failure, I re-engineered the feature set to be time-aware and addressed the severe class imbalance by finding an **optimal prediction threshold** using the Precision-Recall curve. This resulted in a **dramatic performance increase**, creating a model that is both predictive and practical.

### 3. Model Serving via REST API
The final V2 model was wrapped in a REST API using **FastAPI**. The API exposes a `/predict` endpoint that takes a JSON payload of user features and returns a real-time propensity score. FastAPI was chosen for its high speed and automatic generation of interactive documentation (Swagger UI).

### 4. Containerization & Deployment
The entire FastAPI application—including the Python environment, dependencies, and model artifacts—was containerized using **Docker**. This creates a portable, reproducible, and scalable service. The Docker image is deployed as a Web Service on **Render**, which is connected to this GitHub repository for continuous deployment.

## 📈 Results & Performance: V1 vs. V2
The iterative improvement process yielded significant gains, transforming the model from unusable to valuable. The key was trading a small amount of Recall for a massive gain in Precision.

| Metric | Model V1 (Poor Baseline) | Model V2 (Final) | Change | Business Impact |
| :--- | :--- | :--- | :--- | :--- |
| **ROC AUC** | 0.606 | **0.785** | ▲ **+30%** | Much better at distinguishing buyers from non-buyers. |
| **Precision** | 0.0003 | **0.593** | ▲ **+197,567%** | Predictions are now highly reliable and actionable. |
| **Recall** | 0.298 | **0.122** | ▼ **-59%** | A worthwhile trade-off to drastically reduce false positives. |
| **F1 Score** | 0.0006 | **0.203** | ▲ **+33,733%** | The overall model quality is orders of magnitude better. |

## 💻 How to Run This Project

This guide provides instructions to set up the environment, generate the model artifacts locally, and run the API in a Docker container.

#### Prerequisites
- [Git](https://git-scm.com/downloads)
- [Python 3.11+](https://www.python.org/downloads/)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)

#### Step 1: Clone & Set Up Environment
```bash
# Clone this repository
git clone https://github.com/YourUsername/your-repo-name.git
cd your-repo-name

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

#### Step 2: Download Raw Data
1.  Go to the [Retail Rocket E-commerce Dataset on Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset).
2.  Download the dataset and unzip it.
3.  Place **`events.csv`** inside the **`data/`** folder in the project's root.

#### Step 3: Run Training Scripts
This generates the model artifacts needed for the API.
```bash
# From the project root, run the scripts in order
python scripts/process_data.py
python scripts/train_model.py
```

#### Step 4: Build and Run the Docker Container
Make sure Docker Desktop is running.
```bash
# Build the Docker image
docker build -t propensity-api .

# Run the container
docker run -p 8000:8000 propensity-api
```

#### Step 5: Access the Live API
Your API is now running locally!
*   Open your browser and go to **`http://127.0.0.1:8000/docs`** to see the interactive documentation.

## 🔮 Future Improvements
- **Automated Retraining Pipeline:** Implement a workflow orchestration tool like **Prefect** or **GitHub Actions** to run the training scripts on a schedule, creating a fully automated, self-improving system.
- **Model Monitoring:** Build a dashboard (e.g., with Streamlit) to track model performance, data drift, and concept drift over time by logging API requests and predictions.
- **Full CI/CD:** Enhance the GitHub workflow to automatically run tests and linters on every push, only deploying to Render if all checks pass.
