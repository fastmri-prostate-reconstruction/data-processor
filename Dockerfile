FROM pytorch/pytorch:latest
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT ["bash", "entrypoint.sh"]
