# Base image for Python
FROM python:3.10-slim

# Set environment variables to prevent pytecode generation and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy the application code
COPY ./ai_chatbot /app/ai_chatbot

# Copy the requirements file and setup.py for dependency installation
COPY ./requirements.txt /app/
COPY ./setup.py /app/

# Install the dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir .

# Copy the .env file into the container
COPY .env /app/.env

# Expose the application port (e.g. 8000 for FastAPI)
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "ai_chatbot.app:app", "--host", "0.0.0.0", "--port", "8000"]