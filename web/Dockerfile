FROM python:3.6-jessie

RUN pip install --upgrade pip
RUN pip install torch==1.5.0 tqdm uwsgi flask pythainlp deepcut h5py==2.10.0 scikit-learn==0.20.3 tensorflow==2.0.0
RUN pip install flask-cors
RUN pip install numpy -U

ADD . /app
WORKDIR /app

#ENTRYPOINT ["uwsgi", "--ini", "uwsgi.ini"]
CMD exec uwsgi  --http 0.0.0.0:$PORT -p 1 --threads 8 -w wsgi:app
