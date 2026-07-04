# Deploying to DigitalOcean Kubernetes (DOKS)

Deploys the **BGE + Logistic Regression** serving image to a managed Kubernetes
cluster. The manifests live in [`k8s/`](k8s/):

| File | Purpose |
|------|---------|
| `k8s/deployment.yaml` | The pods: image, resources, and startup/readiness/liveness probes |
| `k8s/service.yaml` | `type: LoadBalancer` — DO provisions a managed LB with a public IP |
| `k8s/hpa.yaml` | Optional CPU-based autoscaling (2–5 pods) |
| `k8s/ingress.yaml` | Optional hostname + TLS via ingress-nginx + cert-manager |

## Prerequisites

- [`doctl`](https://docs.digitalocean.com/reference/doctl/how-to/install/) and
  `kubectl` installed, and `doctl auth init` done.
- The image **built and pushed** to your DO Container Registry (see
  [DEPLOY.md](DEPLOY.md) for the build, remembering the trailing `.`):
  ```bash
  doctl registry login
  docker build -t registry.digitalocean.com/teaclipper/is-it-toxic:latest .
  docker push  registry.digitalocean.com/teaclipper/is-it-toxic:latest
  ```

## 1. Create the cluster

Create the cluster with this command

```bash
doctl kubernetes cluster create is-it-toxic \
  --region nyc1 \
  --node-pool "name=pool-1;size=s-1vcpu-2gb;count=1"
```

This also merges the cluster into your kubeconfig and sets it as current. (To do
that manually later: `doctl kubernetes cluster kubeconfig save is-it-toxic`.)

## 2. Let the cluster pull from your registry

Without this, pods fail with `ImagePullBackOff`. This wires your DO Container
Registry credentials into the cluster (adds a pull secret and attaches it to the
`default` ServiceAccount, so no `imagePullSecrets` are needed in the manifests):

```bash
doctl kubernetes cluster registry add is-it-toxic
```

## 3. Setup k8s context

Make sure your kubectl context is set correctly. You should see the k8s context from digital ocean
which looks like `do-nyc1-is-it-toxic`
```sh
$ kubectl config current-context
```

If the current context isn't correct, check which contexts are available:
```sh
kubectl config get-contexts
```

If you don't see your digital ocean context, pull it down with `doctl` 
```sh
doctl kubernetes cluster kubeconfig save is-it-toxic
```

## 3. Deploy

Now deploy:
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

Watch the rollout (pods take ~30–60s to pass the startup probe while the model
loads):

```bash
kubectl rollout status deployment/is-it-toxic
kubectl get pods -l app=is-it-toxic
```

If you're having trouble with the rollout check the pod for issues with:
```bash
kubectl describe pod {podid}
```

## 4. Get the public IP and test

The DO Load Balancer takes a minute or two to get an external IP:

```bash
kubectl get svc is-it-toxic -w
# wait until EXTERNAL-IP is populated, then Ctrl-C
```

```bash
IP=$(kubectl get svc is-it-toxic -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl -X POST http://$IP/is-it-toxic \
     -H "Content-Type: application/json" \
     -d '{"text": "You are the reason this team is failing."}'
```

Interactive docs: `http://$IP/docs`.

## Operating it

```bash
kubectl logs -l app=is-it-toxic -f          # tail logs across pods
kubectl get pods -l app=is-it-toxic         # pod health / restarts
kubectl scale deployment/is-it-toxic --replicas=3
```

**Shipping a new model/image.** Because the manifest uses the `:latest` tag with
`imagePullPolicy: Always`, push a new image then force a rolling restart:

```bash
docker push registry.digitalocean.com/teaclipper/is-it-toxic:latest
kubectl rollout restart deployment/is-it-toxic
kubectl rollout status deployment/is-it-toxic
```

> **Better practice:** tag images immutably (e.g. a git SHA) instead of `:latest`,
> and roll out with `kubectl set image deployment/is-it-toxic api=...:<sha>`. That
> gives you an exact history and easy `kubectl rollout undo`.

## Notes / next steps

- **Memory is the constraint**, not CPU — each pod loads its own copy of PyTorch +
  the BGE model (~1Gi RSS). Size nodes and `replicas` accordingly.
- **HTTPS / a domain:** apply `k8s/ingress.yaml` (after installing ingress-nginx +
  cert-manager) and switch `service.yaml` to `type: ClusterIP`.
- **Cleanup:** `doctl kubernetes cluster delete is-it-toxic` also removes the
  managed Load Balancer it created. (Deleting only the Service also releases the
  LB.)
