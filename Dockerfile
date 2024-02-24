FROM pytorch/pytorch:latest
RUN apt update
RUN apt install -y zip
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT ["bash", "entrypoint.sh"]
