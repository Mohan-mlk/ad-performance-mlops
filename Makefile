.PHONY: install data features train backtest insights drift test lint api dashboard pipeline clean

install:
	pip install -r requirements.txt

data:
	PYTHONPATH=src python -m adperf.data.generate --rows 24000

features:
	PYTHONPATH=src python -m adperf.features.build

train:
	PYTHONPATH=src python -m adperf.models.train

backtest:
	PYTHONPATH=src python -m adperf.models.backtest

insights:
	PYTHONPATH=src python -m adperf.insights.marketing

drift:
	PYTHONPATH=src python -m adperf.monitoring.drift

# Full path from raw data to a served-ready set of artifacts.
pipeline: data features train insights drift

test:
	PYTHONPATH=src pytest tests/ -q

lint:
	ruff check src tests

api:
	PYTHONPATH=src uvicorn adperf.serving.api:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	PYTHONPATH=src streamlit run src/adperf/dashboard/app.py

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

clean:
	rm -rf artifacts/*.joblib artifacts/*.json artifacts/*.csv data/processed/* data/reference/*
