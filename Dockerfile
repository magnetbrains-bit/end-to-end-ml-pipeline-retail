# Dockerfile

# 1. Start from an official Python base image.
FROM python:3.11-slim

# 2. Set the working directory inside the container.
WORKDIR /app

# 3. Copy the requirements file FIRST for layer caching.
COPY requirements.txt .

# 4. Install the Python dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your application code and model artifacts.
COPY ./app .

# 6. Expose the port the application will run on.
EXPOSE 8000

# 7. Define the command to run your application.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "60"]