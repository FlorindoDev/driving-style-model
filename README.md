# Driving Style Model

Analisi dello stile di guida in Formula 1 utilizzando reti neurali (VAE/AutoEncoder) per clusterizzare i dati telemetrici delle curve.

## Installazione

```bash
# 1. Clona il repository
git clone https://github.com/FlorindoDev/RacingDNA
cd RacingDNA

# 2. Crea ambiente virtuale (consigliato)
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Installa dipendenze
pip install -r requirements.txt

# 4. Installa il pacchetto
pip install -e .
```

## Dataset

Il dataset completo utilizzato per il training di questo modello, inclusa la normalizzazione e la struttura dettagliata, è disponibile su Hugging Face:
[**FlorindoDev/f1_corner_telemetry_2024_2025**](https://huggingface.co/datasets/FlorindoDev/f1_corner_telemetry_2024_2025)

> [!NOTE]
> **Credits**: I dati raw telemetrici originali sono stati ottenuti da [**TracingInsights-Archive**](https://github.com/TracingInsights-Archive).


Per dettagli sulla struttura del file `.csv`, la logica di estrazione delle curve, il padding e la normalizzazione, consulta la documentazione dedicata:
[**Vai alla documentazione del Dataset**](data/dataset/README.md)

## Utilizzo

Il progetto offre script pronti all'uso per il training del modello e la valutazione della telemetria.

### 1. Training e Clustering

Lo script `src/scripts/train.py` è il cuore del progetto e gestisce diverse fasi:
1.  **Training**: Addestra il modello (AutoEncoder o VAE) per imparare a rappresentare le curve.
2.  **Encoding**: Utilizza il modello addestrato per comprimere tutte le curve del dataset nello spazio latente.
3.  **Clustering**: Applica l'algoritmo K-Means sullo spazio latente per identificare gruppi (cluster) di stili di guida simili.
4.  **Visualizzazione**: Genera grafici 2D/3D dello spazio latente e dei cluster.

> [!IMPORTANT]
> **Nota sui Pesi Pre-addestrati**
> Per ottenere risultati coerenti utilizzando i pesi inclusi (`src/models/weights/VAE_32z_weights.pth`) e i centroidi (`src/models/weights/kmeans_centroids.npy`), è **fondamentale** utilizzare il **dataset normalizzato** specifico disponibile su Hugging Face:
> [**f1_corner_telemetry_2024_2025/tree/main**](https://huggingface.co/datasets/FlorindoDev/f1_corner_telemetry_2024_2025/tree/main).
>
> L'uso di un dataset diverso o ri-normalizzato localmente renderà i pesi e i centroidi non validi.

**Cosa puoi fare**:
- **Configurare i percorsi**: Modifica `dataset_path`, `load_weights_path` ecc. in `TrainConfig`.
- **Addestrare da zero o caricare**: Imposta `train_model = True` per riaddestrare, `False` per usare pesi salvati.
- **Scegliere il modello**: `use_vae = True` per Variational AutoEncoder, `False` per AutoEncoder standard.
- **Modificare i cluster**: Cambia `num_clusters` per cercare più o meno stili di guida.
- **Gestire i Centroidi**: Usa `save_centroids = True` per salvare i centroidi calcolati in modo da poterli riutilizzare, oppure `load_centroids = True` per caricare centroidi esistenti senza dover rieseguire il clustering (utile se hai già dei cluster definiti e vuoi mantenere coerenza).
- **Attivare grafici**: Imposta su `True` flag come `show_latent_space_2d` o `show_clusters_3d` per vedere i risultati.

```bash
python -m src.scripts.train
```

**Download automatico da Hugging Face**:

Puoi abilitare il download automatico del dataset normalizzato impostando `download_from_hf = True` nella config.

### 2. Valutazione Singola Sessione

Lo script `src/scripts/evaluate.py` serve per applicare il modello addestrato a **nuovi dati** (es. un giro specifico di un pilota).

**Come funziona**:
1.  Carica il file di telemetria grezza (`.json`) specificato.
2.  Usa il file `corners.json` per trovare dove sono le curve nel tracciato.
3.  Estrae e normalizza le curve usando le statistiche del dataset originale.
4.  Proietta ogni curva nello spazio latente e le assegna al cluster più vicino.

**Download automatico da Hugging Face**:

Lo script supporta il download automatico dei dati:
- `download_from_hf = True`: Scarica il dataset normalizzato
- `download_raw_from_hf = True`: Scarica i dati telemetrici grezzi (cartella `2024-main` o `2025-main`)

**Cosa puoi fare**:
- **Analizzare un pilota**: Cambia `telemetry_path` per puntare al file JSON della sessione.
- **Confrontare stili**: Esegui lo script su giri diversi per vedere se lo stile cambia.
- **Verificare l'output**: Lo script stampa per ogni curva l'ID del cluster assegnato.

```bash
python -m src.scripts.evaluate
```

### 3. Visualizzazione Curve

Lo script `src/analysis/curve_visualizer.py` permette di **visualizzare le curve** rilevate da un file di telemetria.

**Cosa puoi fare**:
- Vedere graficamente le curve estratte da una sessione
- Visualizzare le traiettorie delle curve sul tracciato
- Scaricare automaticamente i raw data da HF con `download_from_hf = True`

```bash
python -m src.analysis.curve_visualizer
```

## Struttura Progetto

```
src/
├── analysis/       # Curve detection, preprocessing e curve_visualizer.py
├── models/         # AutoEncoder/VAE, dataset_loader.py (con download HF)
└── scripts/        # train.py, evaluate.py
data/
└── dataset/        # Dataset normalizzati e documentazione
```