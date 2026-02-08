# ReadmeDataset

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
