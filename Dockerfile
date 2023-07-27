# syntax=docker/dockerfile:1
   
FROM python:3.10.5-bullseye

WORKDIR /turbo

COPY . .

RUN apt update && apt install nano
RUN python -m pip install poetry
RUN poetry install && poetry shell

# Create datasets for covid and citibike
RUN ./packaging/create_benchmarks.sh

ENTRYPOINT ["/bin/bash"]