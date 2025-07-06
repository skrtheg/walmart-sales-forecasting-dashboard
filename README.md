# Walmart Sales Forecasting Dashboard

This project is an interactive Streamlit dashboard for Walmart sales forecasting, enhanced with live weather and Google Trends data. It uses machine learning models to predict weekly sales and provides advanced analytics and scenario simulation.

## Features
- Data preprocessing and feature engineering
- Integration of weather and Google Trends data
- Random Forest and Linear Regression models
- Interactive scenario analysis and feature importance visualization
- Anomaly detection and dynamic filtering
- Streamlit dashboard for business insights

## Getting Started

### 1. Clone the repository
```sh
git clone <your-repo-url>
cd walmart_sales_forecasting
```

### 2. Build and Run with Docker Compose
```sh
docker compose up --build
```
- The app will be available at [http://localhost:8501](http://localhost:8501)

### 3. (Alternative) Run Locally Without Docker
```sh
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
- `app.py` - Streamlit dashboard
- `src/` - Core modules (data, models, API integration)
- `data/` - Dataset(s)
- `requirements.txt` - Python dependencies
- `Dockerfile` - For containerization
- `docker-compose.yml` - For multi-container orchestration

## Environment Variables
- Place any API keys or secrets in a `.env` file (excluded from git).

## Notes
- The Dockerfile also installs `uvicorn` (for FastAPI/uv support if you want to add an API endpoint).
- You can extend the service in `docker-compose.yml` to add a backend API or database if needed.

## Learn More
- [Streamlit Docs](https://docs.streamlit.io/)
- [Docker Docs](https://docs.docker.com/)


---

**Happy forecasting!**
