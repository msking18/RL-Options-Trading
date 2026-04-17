# Use the official Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . /app/

# Environment variables for cloud compatibility
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# The container will accept arguments to determine which script to run
# Usage for training: ["python", "-m", "backend.train.train_ppo"]
# Usage for evaluation: ["python", "-m", "backend.train.evaluate_ppo"]
CMD ["python", "-m", "backend.run_pipeline"]
