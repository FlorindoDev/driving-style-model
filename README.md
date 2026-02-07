# Driving Style Model

Analisi dello stile di guida in Formula 1 utilizzando reti neurali (VAE/AutoEncoder) per clusterizzare i dati telemetrici delle curve.

## Installazione

```bash
# 1. Clona il repository
git clone https://github.com/YOUR_USERNAME/driving-style-model.git
cd driving-style-model

# 2. Crea ambiente virtuale (consigliato)
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Installa dipendenze
pip install -r requirements.txt

# 4. Installa il pacchetto
pip install -e .
```

## Utilizzo

```bash
# Training del modello
python -m src.scripts.train

# Valutazione telemetria
python -m src.scripts.evaluate
```

## Struttura Progetto

```
src/
├── analysis/       # Curve detection e preprocessing
├── models/         # AutoEncoder, VAE, weights
└── scripts/        # Script di training e valutazione
data/
└── dataset/        # Dataset normalizzati
```