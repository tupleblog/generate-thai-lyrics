FROM python:3.6-jessie

RUN pip install --upgrade pip
RUN pip install torch deepcut tqdm uwsgi flask
RUN pip install tensorflow==1.5

ADD . /app
WORKDIR /app/web

ENTRYPOINT ["uwsgi", "--ini", "uwsgi.ini"]
