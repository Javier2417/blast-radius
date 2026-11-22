# Pinned image used by the scorer
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

ENV SKIP_INTEGRATION=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    PYTHONPATH=/opt/sqlglot:/app

WORKDIR /app

COPY requirements-runtime.txt /app/requirements-runtime.txt
RUN pip install --no-cache-dir -r /app/requirements-runtime.txt

COPY fetch_sqlglot.py /app/fetch_sqlglot.py
RUN python /app/fetch_sqlglot.py /opt/sqlglot

COPY build_maps.py packer.py patch_engine.py runner.py selector.py main.py /app/
RUN python /app/build_maps.py --sqlglot /opt/sqlglot --out /app/maps --workers 4 \
    && python -m compileall -q /opt/sqlglot /app \
    && chmod -R a+rX /app /opt/sqlglot

CMD ["python", "/app/main.py"]
