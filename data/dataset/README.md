# Readme Dataset

> [!NOTE]
> **Dataset Overview**
> This repository includes:
> - **2024-2025 Curve Dataset**: Includes **Race** and **Qualifying** sessions.
> - **Normalization**: Data is already processed and normalized (see below).
> - **2025 Raw Data**: Raw data from the 2025 season is included.
>
> **Credits**: Raw telemetry data is sourced from [**TracingInsights** on Hugging Face](https://huggingface.co/tracinginsights).

## Dataset Structure

The dataset is a CSV file where each row represents a specific curve taken by a driver in a given lap.

### Identification Columns
The first columns provide metadata for the event and the curve:
- `GrandPrix`: Grand Prix.
- `Session`: Session (e.g., Race).
- `Driver`: Driver (e.g., ALB).
- `Lap`: Lap number.
- `CornerID`: Numeric identifier of the curve.
- `Compound`: Tire compound.
- `TireLife`: Tire life.
- `Stint`: Stint number.

### Telemetry Data (Time Series)
The subsequent columns contain telemetry data sampled within the curve. Each feature has 50 dedicated columns (indexed 0 to 49), corresponding to time samples.

The included features are:
- `speed_[0-49]`: Speed.
- `rpm_[0-49]`: Engine RPM.
- `throttle_[0-49]`: Throttle percentage.
- `brake_[0-49]`: Brake percentage.
- `acc_x`, `acc_y`, `acc_z` `_[0-49]`: Accelerations along the three axes.
- `x`, `y`, `z` `_[0-49]`: Spatial coordinates.
- `distance_[0-49]`: Distance traveled in the lap.
- `time_[0-49]`: Lap timestamp.

### Padding
The dataset uses **padding** to ensure a fixed size of 50 points per curve.
- **Padding Value**: `-1000.0`
- If a curve has fewer than 50 sampled points, the remaining values are filled with `-1000.0`.

---

## Curve Extraction Logic

The logic for identifying and "cutting" curves from the continuous telemetry stream is defined in `src/analysis/CurveDetector.py`. The process aims to isolate driver behavior in each single curve while ensuring structural consistency for the dataset.

The process occurs in 4 main phases:

### 1. Search Window Definition (Spatial Bounding Box)
For each known curve (defined in the `corners.json` file), the algorithm restricts analysis to a specific portion of the track to avoid false positives from other curves.
- The time index corresponding to the **geometric apex point** of the curve (the trajectory point closest to the theoretical apex X,Y coordinates) is identified.
- A search window (`lower_bound` and `upper_bound`) is defined based on distance traveled (`distance`):
  - **Window Start**: Typically 150 meters before the apex (or halfway to the previous curve if very close).
  - **Window End**: Typically 250 meters after the apex (or halfway to the next curve).

### 2. Physical Detection (G-Force Thresholds)
Within this spatial window, the algorithm searches for the actual start and end of the cornering action by analyzing lateral acceleration (`acc_y`).
- **Smoothing**: The `acc_y` signal is first filtered with a moving average (3-sample window) to reduce noise.
- **Curve Entry**: Triggered when `|acc_y_smooth| > 3.0 G`.
- **Curve Exit**: Deactivated when `|acc_y_smooth| < 2.5 G`.
Using two different thresholds (Hysteresis) prevents premature exits due to small fluctuations in G-force during cornering.

### 3. Apex Validation
A sequence of points identified as a "curve" in the previous step is considered valid only if it **includes the apex index**.
- If the algorithm detects a G-force peak that starts and ends *before* the apex, it is discarded.
- This ensures that the extracted data is actually related to traversing the target curve and not to adjacent corrections or maneuvers.

### 4. Cutting and Centering (Clamping)
To standardize the sequence length for neural network input, a hard cut (Clamping) is applied based on the apex position.
- Regardless of what is detected with G-forces, the extracted curve **cannot extend beyond 25 samples** before and after the apex.
- **Start**: If the detected entry is more than 25 samples *before* the apex, it is cut to `apex - 25`.
- **End**: If the detected exit is more than 25 samples *after* the apex, it is cut to `apex + 25`.

### Final Result
The result is a time window of **maximum 50 samples** (slice `[start:end]`) centered on the apex.
- If the curve is very long and fast, it will be truncated at the +/- 25 margins.
- If the curve is short (e.g., slow chicane or quick exit), the window will be fewer than 50 samples. In this case, the **padding** (`-1000.0` values) described above intervenes to fill the empty columns up to 50.

## Data Normalization

Data normalization is handled by the `src/analysis/dataset_normalization.py` script and follows specific logic to preserve the temporal and structural consistency of the curves.

### 1. Grouping by Feature (Grouped Stats)
Columns representing the same physical quantity over time (e.g., from `speed_0` to `speed_49`) are treated as a single group.
- Instead of calculating mean and standard deviation for each individual time column (which could introduce artifacts if the distribution varies along the curve), a **global statistic** is calculated for the entire feature.
- **Example**: The mean of `speed` is calculated on all samples of all curves (all rows, columns 0-49).

### 2. Padding Management
During statistics calculation and normalization, padding values (`-1000.0`) are strictly ignored.
- **Statistics Calculation**: Padding values are replaced with `NaN` before calculating mean and standard deviation to avoid skewing results.
- **Normalized Output**: In the final normalized dataset, padding values are explicitly set to **0.0**. This is important for neural network input (especially if using masking or zero-padding mechanisms).

### 3. Z-Score Normalization
The normalization applied is **Z-Score** (Standardization):
\[ x_{norm} = \frac{x - \mu}{\sigma} \]
Where $\mu$ and $\sigma$ are the global mean and standard deviation of the feature group to which $x$ belongs.

### 4. Categorical Variable Encoding
The `Compound` column undergoes different treatment:
- **One-Hot Encoding** is applied for categories: `HARD`, `INTERMEDIATE`, `WET`, `MEDIUM`, `SOFT`.
- This generates 5 additional binary columns (e.g., `Compound_SOFT`, `Compound_MEDIUM`...) which are not Z-Score normalized (remaining 0/1).

The result is saved in an `.npz` file containing:
- `data`: Matrix of normalized data.
- `mask`: Binary matrix (1=valid, 0=padding).
- `mean`, `std`: Vectors to denormalize data (e.g., to reconstruct original curves).

---

# Readme Dataset (Italiano)

> [!NOTE]
> **Dataset Overview**
> Questa repository include:
> - **Dataset curve 2024-2025**: Include sessioni di **Gara** e **Qualifica**.
> - **Normalizzazione**: I dati sono già processati e normalizzati (vedi sotto).
> - **Dati Raw 2025**: Sono inclusi i dati grezzi della stagione 2025.
>
> **Credits**: I dati raw telemetrici provengono da [**TracingInsights** su Hugging Face](https://huggingface.co/tracinginsights).


## Struttura del Dataset

Il dataset è un file CSV dove ogni riga rappresenta una specifica curva percorsa da un pilota in un determinato giro.

### Colonne Identificative
Le prime colonne forniscono i metadati dell'evento e della curva:
- `GrandPrix`: Gran Premio.
- `Session`: Sessione (es. Race).
- `Driver`: Pilota (es. ALB).
- `Lap`: Numero del giro.
- `CornerID`: Identificativo numerico della curva.
- `Compound`: Mescola delle gomme.
- `TireLife`: Vita delle gomme.
- `Stint`: Numero dello stint.

### Dati Telemetrici (Time Series)
Le colonne successive contengono i dati telemetrici campionati all'interno della curva. Ogni feature ha 50 colonne dedicate (da indice 0 a 49), corrispondenti ai campioni temporali.

Le feature incluse sono:
- `speed_[0-49]`: Velocità.
- `rpm_[0-49]`: Giri motore.
- `throttle_[0-49]`: Percentuale acceleratore.
- `brake_[0-49]`: Percentuale freno.
- `acc_x`, `acc_y`, `acc_z` `_[0-49]`: Accelerazioni lungo i tre assi.
- `x`, `y`, `z` `_[0-49]`: Coordinate spaziali.
- `distance_[0-49]`: Distanza percorsa nel giro.
- `time_[0-49]`: Timestamp del giro.

### Padding
Il dataset utilizza **padding** per garantire una dimensione fissa di 50 punti per curva.
- **Valore di padding**: `-1000.0`
- Se una curva ha meno di 50 punti campionati, i valori rimanenti vengono riempiti con `-1000.0`.

---

## Logica di Estrazione delle Curve

La logica con cui vengono individuate e "tagliate" le curve dal flusso telemetrico continuo è definita in `src/analysis/CurveDetector.py`. Il processo mira a isolare il comportamento del pilota in ogni singola curva, garantendo al contempo una consistenza strutturale per il dataset.

Il processo avviene in 4 fasi principali:

### 1. Definizione della Finestra di Ricerca (Bounding Box Spaziale)
Per ogni curva nota (definita nel file `corners.json`), l'algoritmo restringe l'analisi a una specifica porzione del tracciato per evitare falsi positivi da altre curve.
- Viene identificato l'indice temporale corrispondente al **punto geometrico dell'apice** della curva (il punto della traiettoria più vicino alle coordinate X,Y dell'apice teorico).
- Viene definita una finestra di ricerca (`lower_bound` e `upper_bound`) basata sulla distanza percorsa (`distance`):
  - **Inizio finestra**: Tipicamente 150 metri prima dell'apice (o a metà strada con la curva precedente se molto vicina).
  - **Fine finestra**: Tipicamente 250 metri dopo l'apice (o a metà strada con la curva successiva).

### 2. Rilevamento Fisico (Soglie G-Force)
All'interno di questa finestra spaziale, l'algoritmo cerca l'inizio e la fine effettiva dell'azione in curva analizzando l'accelerazione laterale (`acc_y`).
- **Smoothing**: Il segnale `acc_y` viene prima filtrato con una media mobile (finestra di 3 campioni) per ridurre il rumore.
- **Ingresso Curva**: Si attiva quando `|acc_y_smooth| > 3.0 G`.
- **Uscita Curva**: Si disattiva quando `|acc_y_smooth| < 2.5 G`.
L'uso di due soglie diverse (Isteresi) previene uscite premature dovute a piccole fluttuazioni del segnale G durante la percorrenza.

### 3. Validazione dell'Apice
Una sequenza di punti identificata come "curva" nel passaggio precedente viene considerata valida solo se **include l'indice dell'apice**.
- Se l'algoritmo rileva un picco di G-force che inizia e finisce *prima* dell'apice, viene scartato.
- Questo assicura che i dati estratti siano effettivamente relativi alla percorrenza della curva target e non a correzioni o manovre adiacenti.

### 4. Taglio e Centratura (Clamping)
Per uniformare la lunghezza delle sequenze da dare in pasto alla rete neurale, viene applicato un taglio rigido (Clamping) basato sulla posizione dell'apice.
- Indipendentemente da quanto rilevato con le forze G, la curva estratta **non può estendersi oltre 25 campioni** prima e dopo l'apice.
- **Start**: Se l'ingresso rilevato è oltre 25 campioni *prima* dell'apice, viene tagliato a `apice - 25`.
- **End**: Se l'uscita rilevata è oltre 25 campioni *dopo* dell'apice, viene tagliata a `apice + 25`.

### Risultato Finale
Il risultato è una finestra temporale di **massimo 50 campioni** (slice `[start:end]`) centrata sull'apice.
- Se la curva è molto lunga e veloce, verrà troncata ai margini +/- 25.
- Se la curva è breve (es. chicane lenta o uscita rapida), la finestra sarà inferiore ai 50 campioni. In questo caso, interviene il **padding** (valori `-1000.0`) descritto sopra per riempire le colonne vuote fino ad arrivare a 50.

## Normalizzazione dei Dati

La normalizzazione dei dati è gestita dallo script `src/analysis/dataset_normalization.py` e segue una logica specifica per preservare la consistenza temporale e strutturale delle curve.

### 1. Raggruppamento per Feature (Grouped Stats)
Le colonne che rappresentano la stessa grandezza fisica nel tempo (es. da `speed_0` a `speed_49`) vengono trattate come un unico gruppo.
- Invece di calcolare media e deviazione standard per ogni singola colonna temporale (che potrebbe introdurre artefatti se la distribuzione varia lungo la curva), viene calcolata una **statistica globale** per l'intera feature.
- **Esempio**: La media di `speed` è calcolata su tutti i campioni di tutte le curve (tutte le righe, colonne 0-49).

### 2. Gestione del Padding
Durante il calcolo delle statistiche e la normalizzazione, i valori di padding (`-1000.0`) vengono rigorosamente ignorati.
- **Calcolo Statistiche**: I valori di padding vengono sostituiti con `NaN` prima di calcolare media e deviazione standard, per non falsare i risultati.
- **Output Normalizzato**: Nel dataset finale normalizzato, i valori di padding vengono impostati esplicitamente a **0.0**. Questo è importante per l'input delle reti neurali (soprattutto se si usano meccanismi di mascheramento o zero-padding).

### 3. Z-Score Normalization
La normalizzazione applicata è di tipo **Z-Score** (Standardizzazione):
\[ x_{norm} = \frac{x - \mu}{\sigma} \]
Dove $\mu$ e $\sigma$ sono la media e deviazione standard globali del gruppo di feature a cui appartiene $x$.

### 4. Encoding delle Variabili Categoriche
La colonna `Compound` (mescola) subisce un trattamento diverso:
- Viene applicato **One-Hot Encoding** per le categorie: `HARD`, `INTERMEDIATE`, `WET`, `MEDIUM`, `SOFT`.
- Questo genera 5 colonne binarie aggiuntive (es. `Compound_SOFT`, `Compound_MEDIUM`...) che non vengono normalizzate con Z-Score (rimangono 0/1).

Il risultato è salvato in un file `.npz` che contiene:
- `data`: Matrice dei dati normalizzati.
- `mask`: Matrice binaria (1=valido, 0=padding).
- `mean`, `std`: Vettori per poter denormalizzare i dati (es. per ricostruire le curve originali).
