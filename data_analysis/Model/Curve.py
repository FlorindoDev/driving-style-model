import math
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from utils.utils_for_array import find_closest_value
from scipy.signal import savgol_filter

class Curve:
    def __init__(
        self,
        corner_id: int,
        apex_dist: float,
        lower_bound: float,
        upper_bound: float,
        time: List[float],
        rpm: List[float],
        speed: List[float],
        throttle: List[float],
        brake: List[float],
        distance: List[float],
        acc_x: List[float],
        acc_y: List[float],
        acc_z: List[float],
        x: List[float],
        y: List[float],
        z: List[float],
    ):
        self.corner_id = corner_id
        self.apex_dist = apex_dist
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.time = time
        self.rpm = rpm
        self.speed = speed
        self.throttle = throttle
        self.brake = brake
        self.distance = distance
        self.acc_x = acc_x
        self.acc_y = acc_y
        self.acc_z = acc_z
        self.x = x
        self.y = y
        self.z = z



    #########################################################
    #                   Metodi privati                      #
    #########################################################

    def _normalize_pedal(self, pedal: List[float]) -> List[float]:
        """
        Porta throttle / brake a ~[0,1].
        Se max > 1.5 assumiamo scala 0–100 e dividiamo per 100.
        """
        if not pedal:
            return []
        maxv = max(pedal)
        if maxv == 0:
            return [0.0] * len(pedal)
        if maxv > 1.5:
            return [p / 100.0 for p in pedal]
        return pedal[:]  

    def _savgol_1d(self, data: List[float], window: int = 11, poly: int = 3) -> np.ndarray:
        """
        Applica il filtro Savitzky–Golay per rendere il segnale più liscio
        senza alterarne la forma originale.

        Il filtro funziona così:
        - prende una finestra di punti attorno ad ogni campione
        - ci "fitta" sopra una piccola curva (polinomio di grado `poly`)
        - sostituisce il punto centrale con il valore della curva liscia

        Otteniamo:
        - riduce il rumore
        - mantiene intatta la forma del segnale (picchi, pendenze, curvature)
        - è molto migliore della media mobile perché non appiattisce la curva.

        Params:
        - `window` deve essere dispari (es. 7, 9, 11, ...)
        - `poly` indica quanto complessa può essere la curva locale 
        """
        arr = np.asarray(data, float)

        n = len(arr)
        if n < 3:
            return arr

        # aggiusta window se è troppo lungo o pari
        if window > n:
            window = n
        if window % 2 == 0:
            window -= 1
        if window < poly + 2:
            # troppo pochi punti per un fit decente
            return arr

        return savgol_filter(arr, window_length=window, polyorder=poly, mode="interp")


    def print(self):
        print(f"[!] apex:{self.apex_dist}; start: {self.distance[0]} -> end: {self.distance[-1]}")


    #########################################################
    #                   Metrice semplici                    #
    #########################################################

    # --- tempo ---
    def time_in_curve(self) -> float:
        return self.time[-1] - self.time[0] 

    # --- distanza ---
    def distance_in_curve(self) -> float:
        return self.distance[-1] - self.distance[0] 

     # --- velocità ---
    def speed_average(self) -> float:
        return sum(self.speed) / len(self.speed) 

    def entry_speed_average(self) -> float:
        apex_pilot_idx = find_closest_value(self.distance, self.apex_dist)
        entry_speeds = self.speed[:apex_pilot_idx]
        if len(entry_speeds)  != 0:
            return sum(entry_speeds) / len(entry_speeds) 
        else:
            return 0

    def exit_speed_average(self) -> float:
        apex_pilot_idx = find_closest_value(self.distance, self.apex_dist)
        exit_speeds = self.speed[apex_pilot_idx:]
        if len(exit_speeds)  != 0:
            return sum(exit_speeds) / len(exit_speeds)
        else:
            return 0 
    
    
    def exit_throttle_avg(self):
        apex_idx = find_closest_value(self.distance, self.apex_dist)

        # normalizziamo il pedale gas in [0,1]
        thr = self._normalize_pedal(self.throttle)

        if thr and apex_idx < len(thr):
            post_apex = thr[apex_idx:]
            if post_apex:
                throttle_post_avg = sum(post_apex) / len(post_apex)
            else:
                throttle_post_avg = 0.0
        else:
            throttle_post_avg = 0.0
        return  throttle_post_avg
    
    def apex_speed(self) -> float:
        return min(self.speed) 

    # --- Accelerazioni ---
    def acc_x_max(self) -> float:
        return max(self.acc_x) 

    def acc_x_min(self) -> float:
        return min(self.acc_x) 

    def acc_y_max(self) -> float:
        return max(self.acc_y)

    def acc_resultant(self) -> List[float]:
        return [math.sqrt(ax**2 + ay**2) for ax, ay in zip(self.acc_x, self.acc_y)]

    # --- Frenata ---
    def brake_time(self) -> float:
        total = 0.0
        for i in range(1, len(self.time)):
            if self.brake[i] > 0:
                total += self.time[i] - self.time[i - 1]
        return total

    # --- Acceleratore ---
    def throttle_avg(self) -> float:
        return sum(self.throttle) / len(self.throttle) 

    def throttle_var(self) -> float:
        if not self.throttle:
            return 0.0
        mean = self.throttle_avg()
        return sum((t - mean) ** 2 for t in self.throttle) / len(self.throttle)

    # --- Cambio & motore ---
    def rpm_mean(self) -> float:
        return sum(self.rpm) / len(self.rpm) 

    def rpm_peak(self) -> float:
        return max(self.rpm) 


    #########################################################
    #                Metrice più complesse                  #
    #########################################################


    def curvature_profile(self, smooth_xy: bool = True, smooth_kappa: bool = True) -> List[float]:
        """
        Restituisce una lista kappa[k] (stessa lunghezza di x/y)
        calcolando la curvatura come 1/R del cerchio circoscritto
        a tre punti (k-5, k, k+5).

        Usa Savitzky–Golay per:
        - filtrare x,y (prima del calcolo della curvatura)
        - opzionalmente filtrare anche kappa alla fine
        """

        n = len(self.x)
        if n < 3:
            return [0.0] * n

        x = np.asarray(self.x, float)
        y = np.asarray(self.y, float)


        if smooth_xy:
            x = self._savgol_1d(x, window=11, poly=3)
            y = self._savgol_1d(y, window=11, poly=3)

        kappa = np.zeros(n, dtype=float)

        offset = 5  # distanza tra i 3 punti usati
        for k in range(offset, n - offset):
            x1, y1 = x[k - offset], y[k - offset]
            x2, y2 = x[k],         y[k]
            x3, y3 = x[k + offset], y[k + offset]

            # lati del triangolo
            a = math.hypot(x3 - x2, y3 - y2)
            b = math.hypot(x3 - x1, y3 - y1)
            c = math.hypot(x2 - x1, y2 - y1)

            # area del triangolo (determinante)
            area = 0.5 * abs(
                (x2 - x1) * (y3 - y1) -
                (y2 - y1) * (x3 - x1)
            )

            if area < 1e-9 or a == 0.0 or b == 0.0 or c == 0.0:
                kappa[k] = 0.0
                continue

            R = (a * b * c) / (4.0 * area)
            kappa[k] = 1.0 / R

       
        if smooth_kappa:
            kappa = self._savgol_1d(kappa, window=11, poly=3)

        return kappa.tolist()

    def curvature_mean(self) -> float:
        """
        Faccio una media integrale essendo che le curvature non sono sempre equi distanti 
        ogni curvatura è calcolata su un intervallo di spazio diverso
        """
        kappa = self.curvature_profile()
        n = len(kappa)
        if n < 2 or len(self.distance) != n:
            return 0.0

        num = 0.0
        den = 0.0
        for i in range(1, n):
            ds = self.distance[i] - self.distance[i - 1]
            if ds <= 0:
                continue
            num += abs(kappa[i]) * ds
            den += ds

        return num / den if den > 0 else 0.0


    def energy_input(self) -> float:
        """
        Cosa misura: quanta “energia motore” viene messa dentro la curva.
        formula ∫ throttle * speed dt sul intervallo di tempo 
        throttle * speed = Potenza = F*s
        """

        thr = np.array(self._normalize_pedal(self.throttle))
        spd = np.array(self.speed)
        t   = np.array(self.time)

        if len(t) < 2 or len(thr) != len(t) or len(spd) != len(t):
            return 0.0

        dt = np.diff(t)                         #self.time[i] - self.time[i - 1] cioe il delta per ogni posizione 

        integrand = thr[1:] * spd[1:] * dt      #integrand[i]​=throttle[i]*speed[i]*Δt_i​ integrando sul tempo t

        return float(np.sum(integrand))         # ∑​(thr[i]*speed[i]*dt_i​) calcolo integrale

 
    def energy_lost_brake(self) -> float:
        """
        Cosa misura: quanta energia viene buttata via in frenata.
        formula ∫ brake * (-acc_x_negativa) dt sul intervallo di tempo
        brake * (-acc_x_negativa) = Potenza = F*s

        brake è binario: 0 = non freno, 1 = freno.
        Usiamo acc_x < 0 come intensità della decelerazione
        e integriamo solo dove il freno è premuto.
        """

        ax  = np.asarray(self.acc_x, float)
        brk = np.asarray(self.brake, float)
        t   = np.array(self.time)

        if len(t) < 2 or len(brk) != len(t) or len(ax) != len(t):
            return 0.0

        dt = np.diff(t)                         #self.time[i] - self.time[i - 1] cioe il delta per ogni posizione 

        dec = np.maximum(-ax, 0.0)              #prendo il massimo della decellerazione 
        integrand = brk[1:] * dec[1:] * dt      #integrand[i]​=brk[i]*dec[i]*Δt_i​ integrando sul tempo t

        return float(np.sum(integrand))         # ∑​(brk[i]*dec[i]*Δt_i​) calcolo integrale
 
    def aggressiveness(self) -> float:
        """
        Ritorna un valore in [0,1] che rappresenta l'aggressività del pilota nella curva.

        Componenti:
        - Frenata aggressiva     (decelerazione longitudinale)
        - Apertura gas dopo l’apice
        """

        if not self.time:
            return 0.0

        
        MAX_DECEL = 49.0     # m/s²  (forza longitudinale massima)
        W_BRAKE = 1
        W_GAS   = 2
        
        if self.acc_x:
            min_acc = min(self.acc_x)             
            decel = abs(min_acc)                  
            brake_aggr = min(decel / MAX_DECEL, 1.0)
        else:
            brake_aggr = 0.0

    

        gas_aggr = self.exit_throttle_avg()  

        
        score = (
            W_BRAKE * brake_aggr +
            W_GAS   * gas_aggr
        ) / (W_BRAKE  + W_GAS)

        return max(0.0, min(1.0, score))


    def fluidity(self) -> float:
        """
        Fluidità dello stile di guida basata sulla "dolcezza" dei cambi di:
        - gas normalizzato
        - acc_x (in g)

        Ritorna un valore tra 0 e 1:
        - 0 = guida molto nervosa / a scatti
        - 1 = guida molto fluida / dolce
        """

        if not self.time or len(self.time) < 3:
            return 0.0

        t   = np.asarray(self.time, dtype=float)
        thr = np.asarray(self._normalize_pedal(self.throttle), dtype=float)
        ax  = np.asarray(self.acc_x, dtype=float) / 9.81  # in "g"

        n = min(len(t), len(thr), len(ax))
        if n < 3:
            return 0.0

        t   = t[:n]
        thr = thr[:n]
        ax  = ax[:n]


        dt = np.diff(t)

        # filtro dt non validi, array dove ogni posizione è vera o falsa
        valid = dt > 1e-6
        if not np.any(valid):
            return 0.0

        dt_valid   = dt[valid]
        dthr_dt    = np.diff(thr)[valid] / dt_valid
        dax_dt     = np.diff(ax)[valid]  / dt_valid

        mean_abs_dthr_dt = float(np.mean(np.abs(dthr_dt)))
        mean_abs_dax_dt  = float(np.mean(np.abs(dax_dt)))


        # MAX_DTHR: es. 5.0 → cambio gas 0→1 in 0.2 s (1 / 0.2 = 5)
        MAX_DTHR = 5.0    # [unità gas] per secondo
        # MAX_DAX: es. 3.0 → cambi di 5 g/s considerati "molto nervosi"
        MAX_DAX  = 3.0    # [g] per secondo

        # Nervosità normalizzata in [0,1]
        nerv_thr = min(mean_abs_dthr_dt / MAX_DTHR, 1.0)
        nerv_ax  = min(mean_abs_dax_dt  / MAX_DAX,  1.0)

        # media pesata della nervosità 
        mean_nerv = 0.5 * nerv_thr + 0.5 * nerv_ax

        # Fluidità = opposto della nervosità
        fluidity = 1.0 - mean_nerv


        # safety clamp
        return float(max(0.0, min(1.0, fluidity)))


    def efficiency(self) -> float:
        """
        Efficienza accelerativa:
        (v_out^2 - v_in^2) / dt_curva

        Misura quanto il pilota aumenta l’energia cinetica durante la curva,
        calcolando la variazione di v² (proporzionale alla variazione di energia)
        divisa per il tempo di percorrenza. In pratica indica quanto bene il
        pilota esce dalla curva: valori alti = accelerazione efficace.

        v_in e v_out sono stimati come media delle velocità iniziali/finali
        per rendere la metrica robusta al rumore.

        """
        dt = self.time_in_curve()
        if dt <= 0 or not self.speed:
            return 0.0


        v_in  = self.entry_speed_average()
        v_out = self.exit_speed_average()

        return (v_out**2 - v_in**2) / dt


    def vehicle_stability(self) -> float:
        """
        Plot del Vehicle Stability Index (VSI).

        Pannello superiore:
        - ay misurata vs ay attesa (v²·κ).
        Se coincidono ⇒ veicolo stabile (comportamento ideale).

        Pannello inferiore:
        - |ay - ay_attesa| e sua media.
        Questo errore medio è la base del VSI.

        VSI ∈ [0,1]:
        - alto  ⇒ stabile / coerente con la fisica
        - basso   ⇒ instabile / slip o sovra/sottosterzo
        """

        kappa = self.curvature_profile()

        if not kappa or not self.speed or not self.acc_y:
            return 0.0

        kappa = np.asarray(kappa, float)
        v     = np.asarray(self.speed, float)
        ay    = np.asarray(self.acc_y, float)

        n = min(len(kappa), len(v), len(ay))
        if n < 5:
            return 0.0

        kappa = kappa[:n]
        v     = v[:n]
        ay    = ay[:n]

        v2 = v * v 

        ay_expected = v2 * kappa   #Formula fisica fondamentale del moto circolare. acc_y per quella curvatura

 
        delta_ay = np.abs(ay - ay_expected) #differenza tra quella attessa e quella reale 

        mean_delta = float(np.mean(delta_ay))

        max_ay = 58 #acc_y massima teorica

  
        vsi = 1 - (mean_delta / max_ay)

        return vsi

    def trail_braking_index(self) -> float:
        """
        Trail Braking Index (TBI) in [0,1].

        Misura quanto il pilota frena mentre la vettura è già in appoggio,
        cioè mentre sta generando accelerazione laterale (acc_y).

        Interpretazione:
        - 0.0 = frena solo sul dritto (nessuna frenata mentre è in curva)
        - 1.0 = frena per tutta la durata dell'appoggio laterale
                (trail braking molto aggressivo)

        Formula:
            TBI = ( ∫ brk(t) * |acc_y(t)| dt ) / ( ∫ |acc_y(t)| dt )

        Significato dei termini:
        - |acc_y(t)| misura quanto la vettura sta realmente curvando in ogni istante.
        - brk(t) è 1 quando frena e 0 quando non frena.
        - brk(t) * |acc_y(t)| pesa solo le parti di curva in cui si frena
        mentre c’è appoggio laterale.
        - Il denominatore normalizza il risultato rispetto all'appoggio totale
        della curva, rendendo il valore una percentuale

        In pratica:
        - Numeratore   = quanta frenata avviene "dentro" la curva.
        - Denominatore = quanta curva c’è in totale.
        - TBI          = percentuale della curva percorsa frenando in appoggio.
        """

        if not self.time or not self.acc_y or not self.brake:
            return 0.0

        t   = np.asarray(self.time, float)
        ay  = np.asarray(self.acc_y, float)
        brk = np.asarray(self.brake, float)

        n = min(len(t), len(ay), len(brk))
        if n < 2:
            return 0.0

        t   = t[:n]
        ay  = ay[:n]
        brk = brk[:n]

        dt = np.diff(t)
        

        ay_abs = np.abs(ay[1:])

        num = float(np.sum(brk[1:] * ay_abs * dt))
        den = float(np.sum(ay_abs * dt))

        if den <= 1e-9:
            return 0.0

        return num / den


    def grip_usage(self, max_g: float = 3.0) -> float:
        """
        Grip Usage Index (GUI) in [0,1].

        Calcola il G combinato:
            g_tot = sqrt((acc_x/g)^2 + (acc_y/g)^2)

        acc_x/g e acc_y/g rappresentano le forze longitudinali e laterali
        in unità di "g". g_tot è quindi la forza totale richiesta alle gomme
        in quell'istante (friction circle). Quindi e la richesta di grip del pilotà

        Ritorna:
            mean(g_tot) / max_g   (clampato in [0,1])
        """

        if not self.acc_x or not self.acc_y:
            return 0.0

        ax = np.asarray(self.acc_x, float) / 9.81
        ay = np.asarray(self.acc_y, float) / 9.81

        n = min(len(ax), len(ay))
        if n == 0:
            return 0.0

        ax = ax[:n]
        ay = ay[:n]

        g_tot = np.sqrt(ax * ax + ay * ay)
        mean_g = float(np.mean(g_tot))

        if max_g <= 1e-6:
            return 0.0

        return float(max(0.0, min(mean_g / max_g, 1.0)))


    #########################################################
    #                 Calcolo Stile di guida                #
    #########################################################

    @staticmethod
    def classify_driver_style(curve: "Curve") -> str:
        """
        Classifica la curva in 3 stili:
        - aggressivo
        - medio
        - gestione

        Usa tutte le metriche:
        stability, aggressiveness, fluidity, grip_usage,
        trail_braking, energy_in, energy_lost_brake.
        """

        # --- metriche ---
        stability      = curve.vehicle_stability()      # 1 = stabile, 0 = instabile
        aggressiveness = curve.aggressiveness()         # 0..1
        fluidity       = curve.fluidity()               # 0 = nervoso, 1 = fluido
        energy_in      = curve.energy_input()
        energy_loss    = curve.energy_lost_brake()
        tbi            = curve.trail_braking_index()    # 0..1
        gui            = curve.grip_usage()             # 0..1

        # rapporto energia persa / totale
        eps = 1e-6
        energy_in_safe = max(energy_in, eps)
        brake_ratio = energy_loss / (energy_in_safe + energy_loss + eps)

        # pulizia (fluido + stabile)
        cleanliness = 0.5*stability + 0.5*fluidity
        cleanliness = max(0.0, min(1.0, cleanliness))

        # attacco/aggressività reale
        attack_raw = (
            0.40 * aggressiveness +
            0.25 * gui +
            0.20 * tbi +
            0.15 * brake_ratio
        )
        attack = max(0.0, min(1.0, attack_raw))

        # ---------------- Soglie per i 3 stili ----------------
        # Tarate per far sì che un giro di qualifica risulti più spesso "aggressivo".
        ATTACK_HIGH = 0.45   # sopra → aggressivo
        ATTACK_LOW  = 0.25   # sotto → gestione

        # ---------------- Classificazione ----------------

        if attack >= ATTACK_HIGH:
            return "aggressivo"

        if attack <= ATTACK_LOW:
            return "gestione"

        return "medio"







    #########################################################
    #                   Grafici Per Curva                   #
    #########################################################

    def plot_trajectory_speed(self) -> None:
        """
        Traiettoria XY colorata con la velocità.
        Utile per vedere dove sei più veloce/lento nella curva.
        """
        if not self.x or not self.y or not self.speed:
            print("[plot_trajectory_speed] Dati insufficienti.")
            return

        x = np.array(self.x, dtype=float)
        y = np.array(self.y, dtype=float)
        spd = np.array(self.speed, dtype=float)

        plt.figure()
        sc = plt.scatter(x, y, c=spd)
        plt.colorbar(sc, label="Speed [m/s]")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.title(f"Corner {self.corner_id} – Traiettoria colorata per velocità")
        plt.axis("equal")
        plt.tight_layout()
        plt.show()

    def plot_vehicle_stability(self):
        """
        Disegna i grafici per il Vehicle Stability Index (VSI):

        - sopra: accelerazione laterale misurata vs attesa (v^2 * kappa)
        - sotto: |delta_ay| e sua media

        Nel titolo mostra anche il valore VSI aggregato [0..1] ricavato da vehicle_stability().
        """

        kappa = self.curvature_profile()

        if not kappa or not self.speed or not self.acc_y:
            print("[plot_vehicle_stability] Dati insufficienti.")
            return

        kappa = np.asarray(kappa, float)
        v     = np.asarray(self.speed, float)
        ay    = np.asarray(self.acc_y, float)

        n = min(len(kappa), len(v), len(ay))
        if n < 5:
            print("[plot_vehicle_stability] Troppi pochi punti.")
            return

        kappa = kappa[:n]
        v     = v[:n]
        ay    = ay[:n]

        v2 = v * v

        # se hai distance, uso quella per l'asse X, altrimenti l'indice
        if self.distance and len(self.distance) >= n:
            x_axis = np.asarray(self.distance[:n], float)
            x_label = "Distanza [m]"
        else:
            x_axis = np.arange(n)
            x_label = "Indice campione"

        # Consideriamo solo i punti di curva con un minimo di curvatura e velocità
        mask = (np.abs(kappa) > 1e-6) & (v2 > 1.0)

        if not np.any(mask):
            print("[plot_vehicle_stability] Nessun punto valido per la curva.")
            return

        x_eff   = x_axis[mask]
        kappa   = kappa[mask]
        v2      = v2[mask]
        ay      = ay[mask]

        # 1) Accelerazione laterale attesa
        ay_exp = v2 * kappa

        # 2) Differenza (proxy slip)
        delta_ay = np.abs(ay - ay_exp)

        mean_delta = float(np.mean(delta_ay))

        # uso lo stesso max_ay della metrica aggregata per coerenza

        vsi = self.vehicle_stability()


        # --- PLOT ---
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))

        # 1) ay misurata vs attesa
        ax1.plot(x_eff, ay, label="ay misurata")
        ax1.plot(x_eff, ay_exp, linestyle="--", label="ay attesa (v²·κ)")
        ax1.set_ylabel("ay [m/s²]")
        ax1.set_title(f"Vehicle Stability Index (VSI) ≈ {vsi:.3f}")
        ax1.legend()
        ax1.grid(True)

        # 2) |delta_ay|
        ax2.plot(x_eff, delta_ay, label="|ay - ay_exp|")
        ax2.axhline(mean_delta, linestyle="--",
                    label=f"media |Δay| = {mean_delta:.3f}")
        ax2.set_xlabel(x_label)
        ax2.set_ylabel("|Δay| [m/s²]")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_curvature(self) -> None:
        """
        Profilo di curvatura κ lungo la distanza.
        Usa curvature_profile() e self.distance.
        Nel titolo mostra anche curvature_mean().
        """
        kappa = np.array(self.curvature_profile(), dtype=float)
        dist  = np.array(self.distance, dtype=float)

        if len(kappa) != len(dist) or len(kappa) == 0:
            print("[plot_curvature] Dati insufficienti o mismatch dimensioni.")
            return

        kappa_mean = self.curvature_mean()

        plt.figure()
        plt.plot(dist, kappa)
        plt.xlabel("Distanza [m]")
        plt.ylabel("Curvatura κ [1/m]")
        plt.title(
            f"Corner {self.corner_id} – Profilo di curvatura\n"
            f"curvature_mean ≈ {kappa_mean:.4f} [1/m]"
        )
        plt.tight_layout()
        plt.show()

    def plot_efficiency_profile(self, fraction: float = 0.10, use_time: bool = False) -> None:
        """
        Profilo di efficienza accelerativa:
        mostra la velocità lungo la curva e i segmenti usati per stimare
        v_in e v_out nella metrica di efficienza().
        """
        if not self.speed or not self.distance or not self.time:
            print("[plot_efficiency_profile] Dati insufficienti.")
            return

        spd = np.asarray(self.speed, dtype=float)
        n = len(spd)
        amount = max(1, int(n * fraction))

        # segmenti ingresso/uscita usati per v_in / v_out
        idx_in_end = amount
        idx_out_start = n - amount

        v_in = self.entry_speed_average()
        v_out = self.exit_speed_average()
        eff = self.efficiency()

        x = np.asarray(self.time if use_time else self.distance, dtype=float)
        x_label = "Tempo [s]" if use_time else "Distanza [m]"

        plt.figure()
        plt.plot(x, spd, label="Speed [m/s]", linewidth=2)

        # evidenzia i segmenti di ingresso/uscita usati per la metrica
        plt.axvspan(x[0], x[idx_in_end-1], alpha=0.2, label="zona v_in")
        plt.axvspan(x[idx_out_start], x[-1], alpha=0.2, label="zona v_out")

        # linee orizzontali per v_in e v_out
        plt.axhline(v_in, linestyle="--", label=f"v_in ≈ {v_in:.1f} m/s")
        plt.axhline(v_out, linestyle=":", label=f"v_out ≈ {v_out:.1f} m/s")

        plt.xlabel(x_label)
        plt.ylabel("Velocità [m/s]")
        plt.title(
            f"Corner {self.corner_id} – Profilo efficienza accelerativa\n"
            f"efficiency = (v_out² - v_in²)/dt ≈ {eff:.3f} [m²/s³]"
        )
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    def plot_energy_input_profile(self, use_time: bool = False) -> None:
        """
        Profilo di energy_input:
        mostra l'andamento locale di throttle*speed (≈ “potenza motore”)
        e l'area complessiva corrisponde alla metrica energy_input().
        """
        thr  = np.asarray(self._normalize_pedal(self.throttle), dtype=float)
        spd  = np.asarray(self.speed, dtype=float)
        t    = np.asarray(self.time, dtype=float)
        dist = np.asarray(self.distance, dtype=float)

        if len(t) < 2 or len(thr) != len(t) or len(spd) != len(t):
            print("[plot_energy_input_profile] Dati insufficienti.")
            return

        dt = np.diff(t)
        if np.any(dt <= 0):
            print("[plot_energy_input_profile] dt non valido.")
            return

        # parte “istantanea” (prima di moltiplicare per dt)
        power_like = thr[1:] * spd[1:]
        x_mid_time = 0.5 * (t[1:] + t[:-1])
        x_mid_dist = 0.5 * (dist[1:] + dist[:-1])
        x = x_mid_time if use_time else x_mid_dist
        x_label = "Tempo [s]" if use_time else "Distanza [m]"

        E_in = self.energy_input()

        plt.figure()
        plt.plot(x, power_like, label="thr * speed", linewidth=1.8)
        plt.fill_between(x, 0.0, power_like, alpha=0.2)

        plt.xlabel(x_label)
        plt.ylabel("thr * speed [unità arbitrarie]")
        plt.title(
            f"Corner {self.corner_id} – Profilo energy_input\n"
            f"∫ thr*speed dt ≈ {E_in:.3f} [u.a.]"
        )
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    def plot_energy_lost_brake_profile(self, use_time: bool = False) -> None:
        """
        Profilo di energy_lost_brake:
        mostra l'andamento locale di brake * (-acc_x_negativa)
        (≈ “potenza persa in frenata”) e l'area totale è la metrica
        energy_lost_brake().
        """
        ax   = np.asarray(self.acc_x, dtype=float)
        brk  = np.asarray(self.brake, dtype=float)
        t    = np.asarray(self.time, dtype=float)
        dist = np.asarray(self.distance, dtype=float)

        if len(t) < 2 or len(brk) != len(t) or len(ax) != len(t):
            print("[plot_energy_lost_brake_profile] Dati insufficienti.")
            return

        dt = np.diff(t)
        if np.any(dt <= 0):
            print("[plot_energy_lost_brake_profile] dt non valido.")
            return

        dec = np.maximum(-ax, 0.0)        # solo decelerazione
        loss_inst = brk[1:] * dec[1:]     # parte “istantanea”
        x_mid_time = 0.5 * (t[1:] + t[:-1])
        x_mid_dist = 0.5 * (dist[1:] + dist[:-1])
        x = x_mid_time if use_time else x_mid_dist
        x_label = "Tempo [s]" if use_time else "Distanza [m]"

        E_loss = self.energy_lost_brake()

        plt.figure()
        plt.plot(x, loss_inst, label="brake * dec", linewidth=1.8)
        plt.fill_between(x, 0.0, loss_inst, alpha=0.2)

        plt.xlabel(x_label)
        plt.ylabel("brake * dec [unità arbitrarie]")
        plt.title(
            f"Corner {self.corner_id} – Profilo energy_lost_brake\n"
            f"∫ brake * (-acc_x) dt ≈ {E_loss:.3f} [u.a.]"
        )
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    def plot_aggressiveness_profile(self, use_time: bool = False) -> None:
        """
        Profilo di aggressività:
        - throttle normalizzato
        - derivata d(throttle)/dt (velocità di apertura gas)
        - evidenzia il picco di decelerazione

        Nel titolo viene riportata la metrica aggressiveness().
        """
        if not self.time or not self.throttle or not self.acc_x:
            print("[plot_aggressiveness_profile] Dati insufficienti.")
            return

        t    = np.asarray(self.time, dtype=float)
        dist = np.asarray(self.distance, dtype=float)
        thr  = np.asarray(self._normalize_pedal(self.throttle), dtype=float)
        ax   = np.asarray(self.acc_x, dtype=float)

        if len(t) < 2 or len(thr) != len(t) or len(ax) != len(t):
            print("[plot_aggressiveness_profile] Dati insufficienti (dimensioni).")
            return

        # derivata del gas
        dt = np.diff(t)
        if np.any(dt <= 0):
            print("[plot_aggressiveness_profile] dt non valido.")
            return
        dthr_dt = np.diff(thr) / dt
        x_mid_time = 0.5 * (t[1:] + t[:-1])
        x_mid_dist = 0.5 * (dist[1:] + dist[:-1])
        x = x_mid_time if use_time else x_mid_dist
        x_label = "Tempo [s]" if use_time else "Distanza [m]"

        max_dthr = float(np.max(np.abs(dthr_dt))) if len(dthr_dt) > 0 else 0.0
        ax_min   = float(np.min(ax))
        dec_peak = max(-ax_min, 0.0)
        agg_val  = self.aggressiveness()

        fig, ax1 = plt.subplots()

        # throttle
        x_full = t if use_time else dist
        ax1.plot(x_full, thr, label="Throttle (norm)", linewidth=1.5)
        ax1.set_xlabel(x_label)
        ax1.set_ylabel("Throttle (0–1)", color="black")

        # derivata throttle
        ax2 = ax1.twinx()
        ax2.plot(x, dthr_dt, linestyle="--", label="d(thr)/dt")
        ax2.set_ylabel("d(throttle)/dt [1/s]", color="black")

        # legende combinate
        l1, lab1 = ax1.get_legend_handles_labels()
        l2, lab2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, lab1 + lab2, loc="best")

        plt.title(
            f"Corner {self.corner_id} – Profilo aggressiveness\n"
            f"dec_peak = {dec_peak:.2f} m/s², max|dthr/dt| = {max_dthr:.2f} 1/s, "
            f"aggressiveness = {agg_val:.3f} (0=calmo,1=molto aggressivo)"
        )
        plt.tight_layout()
        plt.show()

    def plot_fluidity_profile(self, use_time: bool = False) -> None:
        """
        Profilo di fluidità:
        mostra throttle e acc_x (scalato) sulla stessa X.
        Serve a vedere visivamente quanto i segnali sono “smooth” o seghettati.
        Nel titolo compare la metrica fluidity() in [0,1].
        """
        if not self.time or not self.throttle or not self.acc_x:
            print("[plot_fluidity_profile] Dati insufficienti.")
            return

        t    = np.asarray(self.time, dtype=float)
        dist = np.asarray(self.distance, dtype=float)
        thr  = np.asarray(self._normalize_pedal(self.throttle), dtype=float)
        ax   = np.asarray(self.acc_x, dtype=float)

        n = min(len(t), len(thr), len(ax))
        if n < 2:
            print("[plot_fluidity_profile] Troppi pochi punti.")
            return

        t    = t[:n]
        dist = dist[:n]
        thr  = thr[:n]
        ax   = ax[:n]

        # scalare acc_x nello stesso range (circa) dei pedali
        ax_abs_max = float(np.max(np.abs(ax))) if np.max(np.abs(ax)) > 0 else 1.0
        ax_scaled = 0.5 + 0.5 * (ax / ax_abs_max)   # porta in ~[0,1]

        x = t if use_time else dist
        x_label = "Tempo [s]" if use_time else "Distanza [m]"

        flu = self.fluidity()

        plt.figure()
        plt.plot(x, thr, label="Throttle (norm)", linewidth=1.5)
        plt.plot(x, ax_scaled, label="acc_x (scalato)", linestyle=":")

        plt.xlabel(x_label)
        plt.ylabel("Valore normalizzato")
        plt.title(
            f"Corner {self.corner_id} – Profilo fluidity\n"
            f"fluidity = {flu:.3f} (0=nervoso, 1=molto fluido)"
        )
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    def plot_trail_braking_profile(self, use_time: bool = False) -> None:
        """
        Profilo di Trail Braking:
        - brake (0/1)
        - |acc_y| (quanta curva stai facendo)
        - evidenzia le zone in cui freni mentre hai carico laterale

        Nel titolo viene mostrato il Trail Braking Index (TBI) in [0,1].
        """
        if not self.time or not self.acc_y or not self.brake:
            print("[plot_trail_braking_profile] Dati insufficienti.")
            return

        t    = np.asarray(self.time, dtype=float)
        dist = np.asarray(self.distance, dtype=float) if self.distance else np.arange(len(t))
        ay   = np.asarray(self.acc_y, dtype=float)
        brk  = np.asarray(self.brake, dtype=float)

        n = min(len(t), len(ay), len(brk), len(dist))
        if n < 2:
            print("[plot_trail_braking_profile] Troppi pochi punti.")
            return

        t    = t[:n]
        dist = dist[:n]
        ay   = ay[:n]
        brk  = brk[:n]

        x = t if use_time else dist
        x_label = "Tempo [s]" if use_time else "Distanza [m]"

        ay_abs = np.abs(ay)

        # maschera: punti in cui stai curvando "abbastanza"
        curve_mask = ay_abs > 0.5  # soglia da tarare
        brake_in_curve = (brk > 0.5) & curve_mask

        tbi = self.trail_braking_index()

        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax1.plot(x, ay_abs, label="|acc_y| [m/s²]", linewidth=1.5)
        ax1.fill_between(x, 0, ay_abs, where=brake_in_curve, alpha=0.3,
                         label="frenata in appoggio")

        ax2 = ax1.twinx()
        ax2.step(x, brk, where="post", linestyle="--", label="brake (0/1)")
        ax2.set_ylabel("Brake")

        ax1.set_xlabel(x_label)
        ax1.set_ylabel("|acc_y| [m/s²]")
        ax1.set_title(
            f"Corner {self.corner_id} – Trail Braking Profile\n"
            f"TBI = {tbi:.3f} (0=solo dritto, 1=frena sempre in appoggio)"
        )

        # legenda combinata
        l1, lab1 = ax1.get_legend_handles_labels()
        l2, lab2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, lab1 + lab2, loc="best")

        ax1.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_grip_usage_profile(self, max_g: float = 3.0, use_time: bool = False) -> None:
        """
        Profilo di Grip Usage:
        - g_tot = sqrt((acc_x/g)^2 + (acc_y/g)^2) lungo la curva
        - linea orizzontale con media(g_tot)
        - nel titolo il Grip Usage Index (GUI) = mean(g_tot) / max_g in [0,1]
        """
        if not self.acc_x or not self.acc_y or not self.time:
            print("[plot_grip_usage_profile] Dati insufficienti.")
            return

        ax   = np.asarray(self.acc_x, dtype=float) / 9.81
        ay   = np.asarray(self.acc_y, dtype=float) / 9.81
        t    = np.asarray(self.time, dtype=float)
        dist = np.asarray(self.distance, dtype=float) if self.distance else np.arange(len(t))

        n = min(len(ax), len(ay), len(t), len(dist))
        if n == 0:
            print("[plot_grip_usage_profile] Troppi pochi punti.")
            return

        ax   = ax[:n]
        ay   = ay[:n]
        t    = t[:n]
        dist = dist[:n]

        g_tot = np.sqrt(ax * ax + ay * ay)
        mean_g = float(np.mean(g_tot))
        gui = self.grip_usage(max_g=max_g)  # usa la metrica

        x = t if use_time else dist
        x_label = "Tempo [s]" if use_time else "Distanza [m]"

        plt.figure(figsize=(10, 5))
        plt.plot(x, g_tot, label="g_tot (G combinato)", linewidth=1.5)
        plt.axhline(mean_g, linestyle="--",
                    label=f"mean(g_tot) ≈ {mean_g:.2f} G")
        plt.axhline(max_g, linestyle=":", label=f"max_g = {max_g:.2f} G")

        plt.xlabel(x_label)
        plt.ylabel("G combinato [g]")
        plt.title(
            f"Corner {self.corner_id} – Grip Usage Profile\n"
            f"GUI = mean(g_tot)/max_g ≈ {gui:.3f}"
        )
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()
