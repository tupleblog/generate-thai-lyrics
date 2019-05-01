FROM python:3.6-jessie

RUN pip install --upgrade pip
RUN pip install torch tqdm uwsgi flask pythainlp
RUN pip install flask-cors
RUN pip install numpy -U

ADD . /app
WORKDIR /app/web

#ENTRYPOINT ["uwsgi", "--ini", "uwsgi.ini"]
CMD exec uwsgi  --http 0.0.0.0:$PORT -p 1 --threads 8 -w wsgi:app
