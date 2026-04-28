# Use an official PyTorch image with CUDA support pre-installed
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /app

# Install system dependencies required for pdf2image (poppler-utils)
RUN apt-get update && apt-get install -y \
    poppler-utils \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the Gradio port
EXPOSE 7860

# Command to run the application
# Note: Ensure your Gradio launch command is set to bind to all interfaces
CMD ["python", "app.py"]
