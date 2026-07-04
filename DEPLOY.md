# Deploying to a DigitalOcean droplet

The container serves the **BGE + Logistic Regression** pipeline only. It bakes in
the pre-trained artifacts and does no training at build time, so builds are fast.

## What the image needs (and a gotcha)

The build copies two things that are **gitignored**, so a plain `git clone` on
the droplet will *not* have them:

- `models/bge-small-en-v1.5/` — the local BGE model
- `models_out/logreg.joblib` — the trained Logistic Regression model
  (produce it locally once with `python classify_logreg.py`)

So you must get these files onto the droplet. Two options below — **rsync** is
the simplest.

## Prerequisites

- A droplet running **Ubuntu 24.04**, **at least 2 GB RAM** (PyTorch + the model
  need headroom; the 1 GB plan will OOM).
- The trained artifact exists locally: run `python classify_logreg.py` once.

## Option A — rsync the project and build on the droplet (simplest)

1. **Create the droplet** and note its IP. Add your SSH key during creation.

2. **Install Docker** on the droplet:
   ```bash
   ssh root@YOUR_DROPLET_IP
   curl -fsSL https://get.docker.com | sh
   ```

3. **Copy the project up, including the gitignored model + artifact.** From your
   local machine (rsync ignores nothing here, so the model comes along):
   ```bash
   rsync -avz --exclude '.git' --exclude 'chroma_db' \
         ./ root@YOUR_DROPLET_IP:/opt/is-it-toxic/
   ```

4. **Build and run** on the droplet:
   ```bash
   ssh root@YOUR_DROPLET_IP
   cd /opt/is-it-toxic
   docker compose up -d --build
   ```

5. **Open the firewall** for HTTP (and keep SSH):
   ```bash
   ufw allow OpenSSH
   ufw allow 80/tcp
   ufw --force enable
   ```

6. **Test it:**
   ```bash
   curl -X POST http://YOUR_DROPLET_IP/is-it-toxic \
        -H "Content-Type: application/json" \
        -d '{"text": "You are the reason this team is failing."}'
   ```
   Interactive docs: `http://YOUR_DROPLET_IP/docs`

## Option B — build locally, push to a registry, pull on the droplet

Good if you don't want build tooling (or the model files) living on the droplet.

1. Build locally (where the model + artifact already exist):
   ```bash
   docker build -t ghcr.io/cwebber314/is-it-toxic:latest .
   ```
2. Push (after `docker login ghcr.io -u cwebber314`, using a PAT with `write:packages`):
   ```bash
   docker push ghcr.io/cwebber314/is-it-toxic:latest
   ```
3. On the droplet, pull and run (public image needs no login; a private one
   needs `docker login ghcr.io` first):
   ```bash
   docker run -d --restart unless-stopped -p 80:8000 \
       ghcr.io/cwebber314/is-it-toxic:latest
   ```

## Operating it

```bash
docker compose logs -f        # tail logs
docker compose ps             # status + health
docker compose restart        # restart
docker compose down           # stop & remove
```

To ship a new model: retrain locally (`python classify_logreg.py`), re-sync (or
rebuild + repush), then `docker compose up -d --build`.

## Notes / next steps

- **HTTPS**: this serves plain HTTP on port 80. For a real domain, put Caddy or
  Nginx in front for TLS, or use a DigitalOcean Load Balancer.
- **Workers**: the container runs a single uvicorn worker. Each worker loads its
  own copy of the model (~hundreds of MB), so only add workers if the droplet has
  the RAM for it.
- **First request** after boot is warmed by a startup hook, so it should be fast.
