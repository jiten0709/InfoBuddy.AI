# Instructions (macOS, minikube):

- Build and push images (set your Docker Hub username and tag)

export DOCKERHUB=jitennn07
export TAG=v1
docker build -t $DOCKERHUB/infobuddy-server-k8s:$TAG ./server
docker push $DOCKERHUB/infobuddy-server-k8s:$TAG
docker build --build-arg NEXT_PUBLIC_SERVER_URL=/api -t $DOCKERHUB/infobuddy-client-k8s:$TAG ./client
docker push $DOCKERHUB/infobuddy-client-k8s:$TAG

- Update image fields in server-deployment.yml and client-deployment.yml to match your repo/tag.

- Apply manifests:

kubectl apply -f k8s/namespace.yml
kubectl apply -f k8s/configmap.yml

# Option A: from file (edit values first) ✅

kubectl apply -f k8s/secret.yml

# Option B: via CLI

kubectl -n infobuddy create secret generic infobuddy-secrets \
 --from-literal=GOOGLE_API_KEY=YOUR_GOOGLE_KEY \
 --from-literal=TAVILY_API_KEY=YOUR_TAVILY_KEY

kubectl apply -f k8s/server-deployment.yml
kubectl apply -f k8s/server-service.yml
kubectl apply -f k8s/client-deployment.yml
kubectl apply -f k8s/client-service.yml
kubectl apply -f k8s/ingress.yml

- If using local images with Minikube, load them into the cluster:

minikube image load jitennn07/infobuddy-client-k8s:v1
minikube image load jitennn07/infobuddy-server-k8s:v1
kubectl -n infobuddy rollout restart deploy/infobuddy-client deploy/infobuddy-server

# cmds for minikube

- minikube start --driver=docker
- minikube tunnel
- minikube status
- kubectl get nodes
- minikube addons enable ingress
- kubectl -n infobuddy get pods
- kubectl -n infobuddy get deploy
- kubectl -n infobuddy logs -l app=infobuddy-client
