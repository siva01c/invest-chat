# Sales Assistant — Development & Deployment Manual

This README focuses on practical steps to run the project locally for development and to deploy it behind an nginx reverse-proxy (nginx-proxy + acme companion) for production.

TL;DR
- For local development: use the provided `docker-compose.override.yml` (maps host ports and enables reload).
- For production behind nginx-proxy: use the main `docker-compose.yml`, create the external `nginx-proxy` network, start the proxy + acme companion, then start services (they expose HTTP on container port 8000).

## Contents
- Prerequisites
- Local development (quick steps)
- Production deployment with nginx-proxy + ACME
- Ports, CORS and common pitfalls
- Useful commands and troubleshooting

## Prerequisites
- Docker and Docker Compose (v2) installed on the host
- A domain name pointing to the host public IP for production (for ACME)
- Optional: Conda or a Python 3.8+ environment for local package-level testing

## Local development
This is the recommended path for active development. It uses `docker-compose.override.yml` that maps container ports to your host so you can access services directly.

1. Configure environment

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY, EMAIL, EMAIL_PWD, etc.
```

2. Start services (override applied automatically)

```bash
docker compose up --build
```

What you get locally
- Assistant web app: http://localhost:5000 (override maps 5000 -> container 8000)
- ChromaDB (host): http://localhost:8001 (override maps host 8001->container 8000)
- Redis: localhost:6379

Notes
- The override runs uvicorn with `--reload` so code changes reload automatically.
- If you prefer to run the app locally as a Python package, you can also create a venv/conda env and run:

```bash
python -m pip install -e .[dev,test]
PYTHONPATH=src uvicorn assistant.api_server:app --reload --host 0.0.0.0 --port 8000
```

## Production deployment (nginx-proxy + ACME companion)
This project is wired to work with `nginx-proxy` (automated vhost generation) and the `acme-companion` for TLS. The main `docker-compose.yml` is designed to run behind that proxy.

High-level steps
1. Create the external proxy network (once on the host):

```bash
docker network create nginx-proxy
```

2. Start the proxy and acme companion (example):

```bash
# from the host, not inside the project compose
# ensure you have volumes for certs / conf mounted as in your proxy compose
docker run -d --name nginx-proxy \
	-p 80:80 -p 443:443 \
	-v /var/run/docker.sock:/tmp/docker.sock:ro \
	-v ./certs:/etc/nginx/certs \
	-v ./vhost.d:/etc/nginx/vhost.d \
	-v ./html:/usr/share/nginx/html \
	nginxproxy/nginx-proxy:1.6-alpine

docker run -d --name nginx-acme --volumes-from nginx-proxy \
	-v /var/run/docker.sock:/var/run/docker.sock:ro \
	-e NGINX_PROXY_CONTAINER=nginx-proxy \
	-e DEFAULT_EMAIL=you@example.com \
	nginxproxy/acme-companion:2.4
```

3. Start the application stack (main compose). The assistant service in `docker-compose.yml` sets `VIRTUAL_HOST` and `VIRTUAL_PORT=8000` so the proxy will route requests for your domain to the assistant container.

```bash
docker compose up -d --build
```

4. Watch proxy logs for cert issuance and vhost creation.

Notes and cautions
- Make sure you do NOT set `VIRTUAL_HOST` for internal-only services like ChromaDB or Redis — otherwise the proxy may expose them publicly.
- Both the assistant and ChromaDB listen on container port 8000 independently; this is fine. The proxy routes by container, not by host port.
- Ensure the `nginx-proxy` network exists and is external in the app compose so the proxy can reach services by Docker network name.

## Ports, CORS and host configuration
- Container internal ports:
	- assistant: 8000 (uvicorn)
	- chromadb: 8000 (internal to chromadb container)
	- redis: 6379 (internal)

- Host mapping when using override (local dev):
	- assistant -> host:5000 (maps to container 8000)
	- chromadb -> host:8001 (maps to container 8000)
	- redis -> host:6379

- CORS: The application will include the `VIRTUAL_HOST` / `LETSENCRYPT_HOST` domain in allowed origins automatically when present. For local dev the override sets `VIRTUAL_HOST=localhost`.

## Troubleshooting
- DNS / ACME failures: Ensure your domain resolves to the host public IP and ports 80/443 are reachable. ACME HTTP-01 requires port 80.
- Proxy not routing: Check that the assistant container is on the `nginx-proxy` network and that the container has `VIRTUAL_HOST` and `VIRTUAL_PORT` env vars set.
- Port collisions on host: If you see an error binding host port 8000, ensure you do not have both assistant and chromadb publishing the same host port. Use the override which maps chromadb to 8001.
- Healthcheck failures: The compose healthchecks use `http://localhost:<container-port>/health` from inside the container. If health fails check the container logs.

## Useful commands

```bash
# Build & run in foreground with local override
docker compose up --build

# Run in background
docker compose up -d --build

# Show running containers and ports
docker ps --format 'table {{.Names}}\t{{.Ports}}'

# Create external network if needed
docker network create nginx-proxy

# Show logs for the proxy
docker logs -f nginx-proxy

# Run tests locally via package
python -m pip install -e .[dev,test]
PYTHONPATH=src python -m pytest -q
```

## Further improvements (suggestions)
- Add a simple `make dev` helper that runs `docker compose up --build` and opens logs.
- Add a short `docs/DEPLOYMENT.md` with more details on renewing certs, or automating with CI/CD.

If you'd like I can add the `make` helper and a short `DEPLOYMENT.md` with example proxy and acme companion commands.

---
End of manual
