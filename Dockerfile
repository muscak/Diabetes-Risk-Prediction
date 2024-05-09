# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Install curl
RUN apt-get update && apt-get install -y curl

# Add the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 10000 available to the world outside this container
ENV PORT=10000
EXPOSE $PORT

# Run main.py when the container launches
CMD uvicorn main:asgi_app --host 0.0.0.0 --port $PORT
