# Smogcast

Forecasts whether tomorrow's average PM2.5 at a monitoring station will go above
35 micrograms per cubic metre, so the city can issue a health alert a day early.

- `app/main.py` is the service. `GET /health` and `POST /predict`.
- `model/` holds the fitted model and its metadata. The service loads both at startup.
- `requirements.txt` lists the exact versions the service needs.
- `train.py` and `make_data.py` made `model/` from `data/readings.csv`. The service
  does not need them, or `data/`, to run.
- `.env` holds this machine's air-quality feed token. It is not committed and must
  never leave this machine.

Run it without Docker, from this folder:

    pip install -r requirements.txt
    uvicorn app.main:app --port 8000
