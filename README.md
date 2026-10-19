\# Heart Disease Prediction — MLOps Project



\## 1. Project Objective



This project demonstrates an end-to-end Machine Learning Operations (MLOps) workflow for classifying heart disease using the UCI Heart Disease dataset.



It covers data preparation, exploratory data analysis, model training, experiment tracking, automated testing, CI/CD, API deployment, containerization, and monitoring.



\*\*Educational use only:\*\* This project is not intended for medical diagnosis or clinical decision-making.



\## 2. Dataset



\- \*\*Source:\*\* \[UCI Machine Learning Repository — Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)

\- \*\*Dataset:\*\* Cleveland Heart Disease dataset

\- \*\*Task:\*\* Binary classification

\- \*\*Target:\*\* Absence or presence of heart disease, according to the project's target conversion logic.



The dataset is downloaded and processed by the data preparation module.



\## 3. Technology Stack



\- Python 3.11

\- pandas and NumPy

\- scikit-learn

\- Matplotlib and Seaborn

\- MLflow

\- FastAPI and Uvicorn

\- pytest and Ruff

\- GitHub Actions

\- Docker

\- Kubernetes with Minikube



\## 4. Environment Setup



Install Anaconda or Miniconda and Git for Windows.



Open Anaconda Prompt and run:



```bash

conda create -n heart-mlops python=3.11 -y

conda activate heart-mlops

cd C:\\Users\\Prakash\\heart-disease-mlops

pip install -r requirements.txt

```



\## 5. Download and Prepare Data



```bash

python -m src.data

```



The processed dataset is saved locally under `data/heart.csv`, according to the data preparation code.



\## 6. Exploratory Data Analysis



Run the EDA module if configured with a command-line entry point:



```bash

python -m src.eda

```



The analysis generates class balance, feature distributions, and correlation visualizations.



\## 7. Train and Evaluate Models



```bash

python -m src.train

```



The training module compares Logistic Regression and Random Forest models using the configured evaluation metrics and cross-validation.



The trained model and evaluation outputs are saved locally by the training script.



\## 8. MLflow Experiment Tracking



Start the MLflow dashboard in a separate terminal:



```bash

conda activate heart-mlops

cd C:\\Users\\Prakash\\heart-disease-mlops

mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

```



Open \[http://127.0.0.1:5000](http://127.0.0.1:5000) to view recorded experiments and metrics.



\## 9. Run the API



Ensure the trained model exists, then run:



```bash

uvicorn src.api:app --reload

```



Open \[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to test the API.



Health endpoint: `GET /health`



Prediction endpoint: `POST /predict`



Monitoring endpoint: `GET /metrics`



\## 10. Automated Tests and Linting



```bash

ruff check src tests

python -m pytest -v

```



\## 11. Docker



Build the image:



```bash

docker build -t heart-disease-api:1.0 .

```



Run the API container:



```bash

docker run --rm -p 8000:8000 heart-disease-api:1.0

```



Verify the API at \[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).



\## 12. Kubernetes with Minikube



Start Minikube and build the image in its Docker environment as described in the deployment instructions.



Apply the Kubernetes configuration:



```bash

kubectl apply -f deployment/deployment.yaml

kubectl get pods

kubectl get services

```



Use the configured service or port forwarding to access the API.



\## 13. CI/CD



GitHub Actions is configured in `.github/workflows/ci.yml`.



The workflow checks out the repository, installs dependencies, runs Ruff, downloads the dataset, trains the models, runs tests, and uploads available artifacts.



This workflow provides continuous integration. It does \*\*not\*\* automatically publish a Docker image or deploy the application. Those steps require an additional deployment workflow.



\## 14. Limitations



\- The dataset is relatively small and may not represent all patient populations.

\- Model metrics do not establish clinical validity.

\- Prediction quality depends on data quality and the chosen model.

\- Local MLflow tracking and deployment are intended for demonstration, not production use.

\- The API and model require further security, validation, monitoring, and clinical evaluation before any real-world use.



