PYTHON ?= python
export MPLBACKEND = Agg
export MLFLOW_DISABLE_AGENT_HINT = 1

.PHONY: setup data clean-data eda notebook train predict mlflow-ui serve \
        docker-build docker-run docker-stop smoke deploy k8s-status k8s-delete load test all

IMAGE ?= cardiorisk-api
TAG ?= latest
API_PORT ?= 8080

setup:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) -m heartrisk.data.download

clean-data: data
	$(PYTHON) -m heartrisk.data.clean

eda: clean-data
	$(PYTHON) -m heartrisk.eda.plots

notebook:
	$(PYTHON) scripts/build_eda_notebook.py
	jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

train: clean-data
	$(PYTHON) -m heartrisk.models.train

predict:
	$(PYTHON) -m heartrisk.models.predict --input samples/patients.json

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

serve:
	uvicorn heartrisk.api.app:app --reload --port $(API_PORT)

docker-build:
	docker build -t $(IMAGE):$(TAG) \
		--build-arg MODEL_VERSION=$$($(PYTHON) -c "import json;print(json.load(open('models/metadata.json'))['model_version'])") \
		--build-arg GIT_SHA=$$(git rev-parse --short HEAD 2>/dev/null || echo local) .

docker-run:
	docker rm -f cardiorisk 2>/dev/null || true
	docker run -d --name cardiorisk -p $(API_PORT):8000 $(IMAGE):$(TAG)

docker-stop:
	docker rm -f cardiorisk

smoke:
	scripts/smoke_test.sh http://localhost:$(API_PORT)

deploy:
	scripts/deploy_minikube.sh

k8s-status:
	kubectl -n cardiorisk get deploy,pods,svc,ingress,hpa,pdb -o wide
	kubectl -n cardiorisk rollout history deployment/cardiorisk-api

k8s-delete:
	kubectl delete -k k8s/overlays/minikube --ignore-not-found

load:
	$(PYTHON) scripts/load_test.py --url http://127.0.0.1:8090 --rps 20 --duration 120

test:
	$(PYTHON) -m pytest -v

all: eda train test
