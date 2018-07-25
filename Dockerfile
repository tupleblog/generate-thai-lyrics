FROM python:3.6-jessie

RUN pip install --upgrade pip
RUN pip install torch deepcut tqdm uwsgi flask

ADD . /app
WORKDIR /app/web
