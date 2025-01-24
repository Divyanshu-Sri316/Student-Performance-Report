FROM python:3.10 
COPY . /app
WORKDIR /app
RUN pip install --upgrade pip
RUN pip install scikit-learn==1.1.3
RUN pip install -r requirements.txt
EXPOSE $PORT
CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app