# 🚀 End-to-End MLOps: Customer Purchase Propensity API 🛍️

This project demonstrates the complete lifecycle of a machine learning model, from data exploration and feature engineering to building a high-performance, containerized REST API for serving real-time predictions.

A key highlight of this project is the **iterative model improvement process**, where an initial baseline model (V1) with poor performance was systematically diagnosed and enhanced to create a high-performance, business-ready model (V2).

**✨ Live API Documentation (Swagger UI):** [https://propensity-api-himanshu.onrender.com/docs](https://propensity-api-himanshu.onrender.com/docs)  
*(Note: The free tier on Render may spin down the service due to inactivity. The first request might take 30-60 seconds to wake the server up.)*

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [The Feature Engineering Strategy](#-the-feature-engineering-strategy)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [The Machine Learning Workflow](#-the-machine-learning-workflow)
- [Results & Performance: V1 vs. V2](#-results--performance-v1-vs-v2)
- [How to Run This Project](#-how-to-run-this-project)
- [Screenshots](#-proof-of-project)
- [Future Improvements](#-future-improvements)

## 🎯 Project Overview

The core business problem is to identify e-commerce users who are most likely to make a purchase in the near future. By identifying these high-propensity users, a business can take targeted actions (e.g., personalized discounts, reminders) to increase conversion rates and revenue.

This project tackles the problem by:
1.  **Engineering predictive features** from raw user behavioral data.
2.  **Training and iterating on a high-performance XGBoost model**.
3.  **Serving the final model** as a live REST API using FastAPI.
4.  **Containerizing the entire application** with Docker for portability and easy deployment to the cloud.

The project uses the [Retail Rocket E-commerce Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset).

## 💡 The Feature Engineering Strategy

The success of the final model hinged on creating features that captured not just *what* a user did, but the **recency and intensity** of their actions. Instead of global aggregates, I engineered a set of **time-windowed features**, which proved to be highly predictive.

For each user, the following features were calculated based on their activity in the **last 1, 7, and 30 days**:
-   `num_views_{n}d`: Total number of items viewed.
-   `num_addtocart_{n}d`: Total number of times an item was added to the cart. This is a strong intent signal.
-   `num_unique_items_{n}d`: The variety of items a user is interested in.
-   `total_events_{n}d`: The overall activity level of the user.

Additionally, two crucial features were created:
-   `days_since_last_event`: Captures user **recency**. A user who was active yesterday is more likely to return than one who was last seen a month ago.
-   `add_to_cart_rate_7d`: The ratio of `addtocart` events to `view` events in the last week. This measures the user's **conversion intent** from browsing to consideration.

These features provided the model with a rich, multi-dimensional view of each user's recent behavior, which was critical for its high performance.

## 🛠️ Tech Stack

- **Data Science & Modeling:**
  - `Python 3.11`
  - `Pandas` & `NumPy` for data manipulation
  - `Scikit-learn` for data splitting and evaluation
  - `XGBoost` for the gradient boosting model
- **API & Engineering:**
  - `FastAPI` for building the high-performance prediction API
  - `Uvicorn` as the ASGI server
  - `Docker` 🐳 for containerization
- **Development & Deployment:**
  - `Google Colab` & `Jupyter` for initial exploration
  - `Visual Studio Code` for local development
  - `Git & GitHub` for version control
  - `Render` ☁️ for cloud deployment

## 📁 Project Structure
The project is organized into distinct modules for clarity and maintainability.
```
.
├── 📁 app/
│   ├── main.py
│   ├── propensity_to_buy_model_v2.pkl
│   └── features_v2.json
├── 📁 data/
├── 📁 scripts/
│   ├── process_data.py
│   └── train_model.py
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

## 🔄 The Machine Learning Workflow

1.  **Data Processing & Feature Engineering:** The raw data was processed to create the time-windowed features described above. This logic is encapsulated in `scripts/process_data.py`.
2.  **Model Training & Iteration:** An **XGBoost Classifier** was trained. After an initial poor-performing baseline (V1), the model was retrained on the advanced features (V2) and the prediction threshold was optimized, leading to a massive performance increase.
3.  **API Service:** The final V2 model was wrapped in a **FastAPI** REST API that exposes a `/predict` endpoint.
4.  **Containerization & Deployment:** The entire application was containerized using **Docker** and deployed as a Web Service on **Render**, which is connected to this GitHub repository for continuous deployment.

## Flowchart
flowchart LR
    A[Data Source: Kaggle] --> B[Data Processing & Feature Engineering<br/>Python/Pandas]
    B --> C[Model Training<br/>XGBoost]
    C --> D[(Model Artifact<br/>.pkl)]
    D --> E[API Service<br/>FastAPI]
    E --> F[Containerization<br/>Docker]
    F --> G[Cloud Deployment<br/>Render]
    
    %% Styling
    classDef dataSource fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    classDef processing fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef model fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef artifact fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef deployment fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    
    class A dataSource
    class B,C processing
    class E model
    class D artifact
    class F,G deployment

## 📈 Results & Performance: V1 vs. V2
The iterative improvement process was critical. The key was trading a small amount of Recall for a massive gain in Precision, making the model's predictions highly reliable.

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
git clone https://github.com/magnetbrains-bit/end-to-end-ml-pipeline-retail.git
cd end-to-end-ml-pipeline-retail

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

## Screenshots

1.**Render deployment log**

![Screenshot 2025-06-14 203259](https://github.com/user-attachments/assets/d49db62f-bf11-46b9-b301-f1498c61db1e)

2.**Swagger UI** 

![Screenshot 2025-06-14 205134](https://github.com/user-attachments/assets/ddb5232a-69b2-4cc8-bb80-1d6dc9582761)

3.**Profile 1: The "Hot Lead" 🔥:**

{
  "features": {
    "total_events_30d": 50,
    "num_views_30d": 40,
    "num_addtocart_30d": 10,
    "num_unique_items_30d": 20,
    "total_events_7d": 25,
    "num_views_7d": 20,
    "num_addtocart_7d": 5,
    "num_unique_items_7d": 10,
    "total_events_1d": 8,
    "num_views_1d": 7,
    "num_addtocart_1d": 1,
    "num_unique_items_1d": 4,
    "days_since_last_event": 0,
    "add_to_cart_rate_7d": 0.25
  }
}

![Screenshot 2025-06-14 203559](https://github.com/user-attachments/assets/fcf2ec58-3f43-4ecc-9d84-e3a941a5b743)

4.**Profile 2: The "Lapsed User" 😴**

{
  "features": {
    "total_events_30d": 20,
    "num_views_30d": 18,
    "num_addtocart_30d": 2,
    "num_unique_items_30d": 15,
    "total_events_7d": 0,
    "num_views_7d": 0,
    "num_addtocart_7d": 0,
    "num_unique_items_7d": 0,
    "total_events_1d": 0,
    "num_views_1d": 0,
    "num_addtocart_1d": 0,
    "num_unique_items_1d": 0,
    "days_since_last_event": 20,
    "add_to_cart_rate_7d": 0.0
  }
}

![Screenshot 2025-06-14 203712](https://github.com/user-attachments/assets/2822cace-967a-49e7-b51c-b667244e266f)

5.**Profile 3: The "Minimalist" (Edge Case)**

{
  "features": {
    "total_events_30d": 2,
    "num_views_30d": 1,
    "num_addtocart_30d": 1,
    "num_unique_items_30d": 1,
    "total_events_7d": 2,
    "num_views_7d": 1,
    "num_addtocart_7d": 1,
    "num_unique_items_7d": 1,
    "total_events_1d": 0,
    "num_views_1d": 0,
    "num_addtocart_1d": 0,
    "num_unique_items_1d": 0,
    "days_since_last_event": 5,
    "add_to_cart_rate_7d": 1.0
  }
}

![Screenshot 2025-06-14 203741](https://github.com/user-attachments/assets/796b83b0-765b-4a5c-a3bf-d2d7f80bd8c6)

## 🔮 Future Improvements
- **Automated Retraining Pipeline:** Implement a workflow orchestration tool like **Prefect** or **GitHub Actions** to run the training scripts on a schedule, creating a fully automated, self-improving system.
- **Model Monitoring:** Build a dashboard (e.g., with Streamlit) to track model performance and data drift over time by logging API requests and predictions.
- **Full CI/CD:** Enhance the GitHub workflow to automatically run tests and linters on every push, only deploying to Render if all checks pass.
