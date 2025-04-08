# Use a slim Python image to keep it lightweight
FROM python:3.10-slim

# Set working directory in the container
WORKDIR /app

# Copy only the backend folder contents
COPY backend/ .

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn requests pandas

# Expose port 8000 for the app
EXPOSE 8000

# Start FastAPI server with main.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
