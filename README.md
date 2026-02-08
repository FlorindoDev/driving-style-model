# Driving Style Model

Analisi dello stile di guida in Formula 1 utilizzando reti neurali (VAE/AutoEncoder) per clusterizzare i dati telemetrici delle curve.

## Installazione

```bash
# 1. Clona il repository
git clone https://github.com/FlorindoDev/driving-style-model.git
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

### 2. Valutazione Singola Sessione

Lo script `src/scripts/evaluate.py` serve per applicare il modello addestrato a **nuovi dati** (es. un giro specifico di un pilota).

**Come funziona**:
1.  Carica il file di telemetria grezza (`.json`) specificato.
2.  Usa il file `corners.json` per trovare dove sono le curve nel tracciato.
3.  Estrae e normalizza le curve usando le statistiche del dataset originale.
4.  Proietta ogni curva nello spazio latente e le assegna al cluster più vicino.

**Dove trovare i dati per la valutazione**:
Per provare lo script, hai bisogno di due file per ogni sessione: il file della telemetria (es. `4_tel.json` per Norris) e il file delle curve (`corners.json`) per quel tracciato.

Puoi scaricarli da:
- **Hugging Face**: Nella cartella [**data/2025-main** del dataset](https://huggingface.co/datasets/FlorindoDev/f1_corner_telemetry_2024_2025/tree/main/data/2025-main) trovi esempi già organizzati.
- **TracingInsights Archive**: Puoi scaricare i dati grezzi direttamente dalla [repository originale 2025](https://github.com/TracingInsights-Archive/2025).
    - *Nota*: Assicurati di scaricare anche il file `corners.json` corrispondente al Gran Premio.

**Cosa puoi fare**:

- **Analizzare un pilota**: Cambia `telemetry_path` per puntare al file JSON della sessione che ti interessa.
- **Confrontare stili**: Esegui lo script su giri diversi per vedere se lo stile di guida (cluster ID) cambia.
- **Verificare l'output**: Lo script stampa per ogni curva l'ID del cluster assegnato e una statistica finale (es. "Dominant style: Cluster 2").

```bash
python -m src.scripts.evaluate
```

## Struttura Progetto

```
src/
├── analysis/       # Curve detection (CurveDetector.py) e preprocessing
├── models/         # Definizioni AutoEncoder/VAE e gestione dataset
└── scripts/        # train.py, evaluate.py
data/
└── dataset/        # Dataset normalizzati e documentazione (ReadmeDataset.md)
```