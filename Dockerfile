FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# unprivileged user
RUN useradd --uid 1000 --create-home meatproxy

WORKDIR /home/meatproxy/app

COPY pyproject.toml README.md ./
COPY src ./src

# app deps (incl. github-copilot-sdk)
RUN pip install --no-cache-dir .

# bake the Copilot CLI runtime into the image (no runtime download needed).
# Runs as the app user so the cache lands in /home/meatproxy/.cache.
USER meatproxy
RUN python -m copilot download-runtime

# persistent state: SQLite conversations + Copilot session state / login.
# Mount a volume at /meatproxy.yaml (config) and at the Copilot home
# (default: /home/meatproxy/.copilot) so `meatproxy copilot login --yes`
# persists across container runs.
ENV MEATPROXY_CONFIG=/meatproxy.yaml
VOLUME ["/data"]

EXPOSE 8700

USER meatproxy
WORKDIR /home/meatproxy
ENTRYPOINT ["meatproxy", "serve"]
