# Pinned image used by the scorer
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

WORKDIR /app

# Pre-computed transitive import, fixture, and dialect maps (95 KB)
COPY maps.json /app/maps.json
COPY selector.py /app/selector.py

# Ensure world-readable for user nobody (uid 65534)
RUN chmod -R a+r /app

CMD ["python", "/app/selector.py"]
