FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy the application files
COPY . /app

# Upgrade pip and install dependencies
RUN pip install --upgrade pip \
    && pip install numpy==1.23.5 \
    && pip install -r requirements.txt \
    && pip install --no-binary scikit-learn scikit-learn

# Expose the port for the application
EXPOSE $PORT

# Start the application
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app
