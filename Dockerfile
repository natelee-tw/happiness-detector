FROM continuumio/miniconda3

WORKDIR /app

COPY conda.yml .

RUN apt-get install -y libgl1-mesa-dev

RUN conda env create -f conda.yml

RUN conda env update -f conda.yml -n base

COPY src/. src/.
COPY app.py .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py"]