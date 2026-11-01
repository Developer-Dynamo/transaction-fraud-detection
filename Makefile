.PHONY: help install train check-features api seed stream dashboard sim-data lint docker-up docker-stream docker-down clean

PY ?= python

help:           ## Liste les commandes
	@grep -E '^[a-z-]+:.*##' Makefile | sed 's/:.*##/ :/'

install:        ## Installe les dépendances
	$(PY) -m pip install -r requirements.txt

train:          ## Entraîne le modèle (data/fraudTrain_clean.csv -> models/ + reports/)
	$(PY) src/Train_ML.py --data data/fraudTrain_clean.csv

check-features: ## Vérifie que features_builder.py reproduit le CSV de features de référence
	$(PY) src/features_builder.py data/fraudTrain_features.csv

api:            ## API de scoring (port 8001)
	$(PY) src/API_Score.py

seed:           ## Rejoue l'historique des cartes (API démarrée)
	$(PY) src/historique.py --csv data/sample/History.csv

stream:         ## Flux de transactions simulées vers l'API
	$(PY) src/Ingestion.py

dashboard:      ## Dashboard (port 8002)
	$(PY) src/dashboard.py

sim-data:       ## Génère un jeu mondial simulé étiqueté dans data/sim_world/
	$(PY) src/simulate_world.py --out data/sim_world

lint:           ## Erreurs de syntaxe / noms non définis
	ruff check src --select E9,F63,F7,F82

docker-up:      ## API + dashboard (Docker Compose)
	docker compose up --build

docker-stream:  ## Ajoute le flux simulé
	docker compose --profile stream up ingestion

docker-down:    ## Arrête les conteneurs
	docker compose down

clean:          ## Supprime les caches Python
	find . -type d -name "__pycache__" -not -path "./venv/*" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
