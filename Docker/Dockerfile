FROM python:3.9
WORKDIR /Karpathy-Challenge
USER root


COPY requirements.txt .
RUN pip3 install --user --no-cache-dir -r requirements.txt

COPY packages.txt .
RUN apt-get update && \
    xargs -a packages.txt apt-get install -y

COPY . .
ENV PATH=/root/.local/bin:$PATH

CMD ["streamlit", "run", "streamlit_app.py"]

EXPOSE 8501