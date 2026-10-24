FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 SPORTS_LAB_DB=/data/sports_lab.sqlite3
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd -m app && mkdir -p /data && chown app /data
USER app
VOLUME /data
EXPOSE 8501
HEALTHCHECK CMD python -c "import os,urllib.request; urllib.request.urlopen('http://localhost:%s/_stcore/health' % os.environ.get('PORT','8501'))" || exit 1
CMD streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}
