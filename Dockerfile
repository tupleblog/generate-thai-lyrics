FROM python:3.6-jessie

RUN pip install --upgrade pip
RUN pip install torch tqdm uwsgi flask pythainlp pickle
RUN pip install flask-cors

ADD . /app
WORKDIR /app/web

ENTRYPOINT ["uwsgi", "--ini", "uwsgi.ini"]
