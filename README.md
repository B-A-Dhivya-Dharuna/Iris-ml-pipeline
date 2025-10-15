# 🐳 Iris ML Pipeline Deployment Project
This repository contains a complete Machine Learning pipeline that trains a K-Nearest Neighbors (KNN) model on the Iris dataset. It showcases a full MLOps workflow—from data preparation to automated Docker image deployment using GitHub Actions.

## 🚀 Project Overview
This project simulates a typical MLOps pipeline:

**📊 Data Preparation** Loads the Iris dataset and splits it into training and testing sets: data/iris_train.csv, data/iris_test.csv

**🧠 Model Training** Trains a KNeighborsClassifier and evaluates its performance.

**💾 Model Persistence** Saves the trained model as a pickle file: models/iris_knn_model.pkl

**📦 Containerization Packages** the entire application into a Docker image using a custom Dockerfile.

**⚙️ CI/CD Automation** Uses GitHub Actions to automatically build and push the Docker image to Docker Hub on every push to the main branch.

**📁 Repository Structure**
Code
├── .github/
│   └── workflows/
│       └── docker-publish.yml     # GitHub Actions workflow
├── Dockerfile                     # Defines container environment
├── iris.py                        # ML pipeline script
├── requirements.txt               # Python dependencies
├── data/
│   ├── iris_train.csv             # Training data
│   └── iris_test.csv              # Testing data
├── models/
│   └── iris_knn_model.pkl         # Saved model
└── README.md                      # This file
## 🛠️ Local Setup & Execution
**🔧 Prerequisites**
-> Python 3.9+

-> Docker Desktop

-> Git

**🐳 Build the Docker Image**
bash
docker build -t iris_image .

**▶️ Run the Container**
bash
docker run iris_image

**Expected Output:**
Code
✅ Training data saved to: data/iris_train.csv (Rows: 105)
✅ Test data saved to: data/iris_test.csv (Rows: 45)
✅ Trained model saved as: models/iris_knn_model.pkl

## 🐙 Automated Deployment with GitHub Actions
**🔐 Secrets Configuration**
To enable Docker Hub authentication, configure these secrets in your GitHub repo:

Secret Name	Description
--------------------------------------------------------------
|DOCKER_USERNAME |	Your Docker Hub username                 |
|DOCKER_PASSWORD |	Docker Hub Access Token (not password)   |
--------------------------------------------------------------

**Location:** GitHub → Settings → Secrets and variables → Actions

**⚙️ Workflow Details**
-> Defined in .github/workflows/docker-publish.yml
-> Triggered on every push to main
-> Uses docker/build-push-action to build and push the image

**📦 Final Image Location**
bash
docker pull dhivyadharuna/iris-ml-pipeline:latest

Or view it on Docker Hub

## 📝 Key Files
iris.py
Contains two main functions:
-> download_and_split_data() Loads and splits the Iris dataset into CSV files.
-> train_and_evaluate_model() Trains a KNN model and saves it as a pickle file.

Dockerfile:-

FROM python:3.9-slim-buster
WORKDIR /app 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "iris.py"]

requirements.txt:-

Code
pandas
scikit-learn
joblib
