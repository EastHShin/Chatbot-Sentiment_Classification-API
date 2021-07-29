FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel

RUN pip install flask transformers requests torch
WORKDIR /app

COPY . .

EXPOSE 5000

CMD ["flask", "run", "--host", "0.0.0.0"]