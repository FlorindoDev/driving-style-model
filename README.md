# RacingDNA

Analisi dello stile di guida in Formula 1 utilizzando reti neurali (VAE/AutoEncoder) per clusterizzare i dati telemetrici delle curve.

## Il Progetto

Il cuore del progetto consiste nell'analizzare le telemetrie di Formula 1 per identificare e classificare lo stile di guida dei piloti in curva. Il flusso di lavoro è strutturato come segue:

1.  **Estrazione Curve** (`src/analysis/CurveDetector.py`):
    Le telemetrie grezze vengono analizzate per estrarre i segmenti relativi alle curve. Questo avviene utilizzando:
    -   **Bound spaziali**: Definizione di un'area di interesse attorno all'apice della curva basata sulla mappa del circuito (`corners.json`).
    -   **Finestre e Smoothing**: Applicazione di una *moving average* sull'accelerazione laterale (`acc_y`) per ridurre il rumore.
    -   **Thresholding con Isteresi**: L'ingresso in curva è rilevato quando la forza G supera una soglia (`ACC_ENTER_THR`, default 3.0 G), mentre l'uscita avviene quando scende sotto una soglia inferiore (`ACC_EXIT_THR`, default 2.5 G), garantendo stabilità nella rilevazione.

2.  **Costruzione e Normalizzazione Dataset** (`src/analysis/dataset_normalization.py`):
    Le curve estratte vengono raccolte in un dataset. Viene applicata una **Z-score normalization** (sottrazione della media e divisione per la deviazione standard) calcolata sull'intero dataset per rendere i dati omogenei e adatti all'addestramento della rete neurale.

3.  **Modellazione con VAE** (`src/models/VAE.py`, `src/models/AutoEncoder.py`):
    Una rete neurale di tipo **VAE (Variational AutoEncoder)** (o AutoEncoder standard) viene addestrata sul dataset normalizzato. Il VAE impara a comprimere i dati complessi della curva (velocità, frenata, accelerazione, traiettoria) in una rappresentazione compatta chiamata **spazio latente**.

4.  **Clustering e Stile di Guida** (`src/scripts/train.py`):
    Lo spazio latente generato dall'Encoder viene suddiviso in **4 cluster** utilizzando l'algoritmo **K-Means**.  questi cluster rappresentano diversi stile di guida (es. aggressivo , gestione, costante, ecc.), permettendo di etichettare automaticamente il comportamento del pilota in ogni curva.

### Input e Output per Script

Di seguito i dettagli degli input e output specifici per ogni script principale del progetto.

#### 1. `src/scripts/train.py` (Training e Clustering)
*   **Input**:
    *   **Dataset Normalizzato**: File `.npz` contenente le curve processate (scaricabile automaticamente da Hugging Face).
    *   **Configurazione**: Parametri definiti in `TrainConfig` (es. `latent_dim`, `num_clusters`, `epochs`).
    *   **Pesi Pre-addestrati** (opzionale): File `.pth` se si sceglie di non riaddestrare il modello.
    *   **Centroidi** (opzionale): File `.npy` se si vuole caricare una clusterizzazione esistente.
*   **Output**:
    *   **Modello Addestrato**: File dei pesi `.pth` (se `train_model = True`).
    *   **Centroidi K-Means**: File `.npy` salvato dopo il clustering.
    *   **Visualizzazioni**: Grafici 2D/3D dello spazio latente e dei cluster.
    *   **Statistiche Cluster**: Stampa a console della distribuzione e caratteristiche dei cluster.

#### 2. `src/scripts/evaluate.py` (Valutazione Telemetria)
*   **Input**:
    *   **Telemetria Grezza**: File `.json` di una sessione (es. `1_tel.json`).
    *   **Mappa Curve**: File `corners.json` relativo al circuito.
    *   **Modello**: Pesi del VAE/AutoEncoder (`.pth`).
    *   **Dataset Normalizzato**: Necessario per caricare le statistiche (media/std) usate per la normalizzazione.
    *   **Centroidi**: File `.npy` per assegnare i cluster.
*   **Output**:
    *   **Classificazione**: Stampa a console della sequenza curva-per-curva con il cluster assegnato e lo stile di guida predominante.
    *   **Analisi**: Percentuali di utilizzo dei vari stili di guida nella sessione.

#### 3. `src/analysis/curve_visualizer.py` (Visualizzazione Curve)
*   **Input**:
    *   **Telemetria Grezza**: File `.json` della sessione.
    *   **Mappa Curve**: File `corners.json` del circuito.
*   **Output**:
    *   **Grafici Segnali**: Plot dei canali telemetrici (Accelerazione Laterale, Freno, Acceleratore) per ogni curva rilevata.
    *   **Traiettorie**: Visualizzazione 2D della traiettoria percorsa in ogni curva rispetto al tracciato.
    *   **Debug**: Verifica visiva della correttezza dei bound e del rilevamento curve.

## Installazione

**IDE utilizzato per lo sviluppo**: Visual Studio Code

```bash
# 1. Clona il repository
git clone https://github.com/FlorindoDev/RacingDNA.git
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
> **Nota sul Download Automatico**:
> Il codice include già dei flag (`download_from_hf = True`) nei file di configurazione (`train.py`, `evaluate.py`) attivati di default. Questo significa che avviando gli script, il dataset **corretto e normalizzato** verrà scaricato automaticamente da Hugging Face, garantendo la totale compatibilità con i pesi forniti.
>
> **Nota sui Pesi Pre-addestrati**:
> Se si decidesse di non usare il download automatico, per ottenere risultati coerenti utilizzando i pesi inclusi (`src/models/weights/VAE_32z_weights.pth`) e i centroidi (`src/models/weights/kmeans_centroids.npy`), è **fondamentale** utilizzare manualmente il dataset specifico disponibile qui:
> [**f1_corner_telemetry_2024_2025/tree/main**](https://huggingface.co/datasets/FlorindoDev/f1_corner_telemetry_2024_2025/tree/main).
>
> L'uso di un dataset diverso o ri-normalizzato localmente renderà i pesi e i centroidi non validi.
>
> **Significato dei Cluster (Configurazione Default)**:
> Utilizzando i pesi e i centroidi inclusi, i 4 cluster identificati hanno il seguente significato:
> - **Cluster 0**: Mantiene un passo veloce.
> - **Cluster 1**: Gestione (Saving).
> - **Cluster 2**: Spinge (Pushing).
> - **Cluster 3**: Mantiene un passo, ma lento.

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