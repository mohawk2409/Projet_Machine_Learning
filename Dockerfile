# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requierements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requierements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose port 8000 for Streamlit dashboard
EXPOSE 8000

# Command to run the Streamlit dashboard
CMD ["streamlit", "run", "src/dashboard_clustering.py", "--server.port=8000", "--server.address=0.0.0.0"]