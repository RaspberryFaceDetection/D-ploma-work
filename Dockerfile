FROM python:3.6

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN  pip install cmake==3.17.2 && pip install -r requirements.txt

COPY . /app
