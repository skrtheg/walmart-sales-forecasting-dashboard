# Use official Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if needed for numpy, pandas, etc.)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install uvicorn (for uv package, if you want to run FastAPI or similar)
RUN pip install "uvicorn[standard]"

# Copy the rest of your code
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Set environment variables for Streamlit
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ENABLECORS=false

# Default command: run Streamlit app
CMD ["streamlit", "run", "app.py"]
