import math
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from utils.utils_for_array import find_closest_value


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


    def print(self):
        print(f"[!] apex:{self.apex_dist}; start: {self.distance[0]} -> end: {self.distance[-1]}")


    def _time_apex(self):
        apex_indx = find_closest_value(self.distance, self.apex_dist)
        return self.time[apex_indx] 

    def _time_entry(self):
        return self.time[0]

    def _t_brake_on(self):
        apex_indx = find_closest_value(self.distance, self.apex_dist)

        tmp = self.brake[0:apex_indx]

        get_start = False
        start_brake=None
        i = 0

        for brake in tmp:
            if brake == 1:
                if not get_start:
                    get_start = True
                    start_brake = i
            else:
                if get_start and i - start_brake > 3:
                        return self.time[start_brake] 
                else:
                    get_start = False
            i+=1

        if start_brake == None or i - start_brake < 3:
            return None
        else:
            return self.time[start_brake] 
            
    def _time_throttle_start(self, threshold=0.20):
        apex_indx = find_closest_value(self.distance, self.apex_dist)
        for i in range(apex_indx, len(self.throttle)):
            if self.throttle[i] > threshold:
                return self.time[i]
        return None

    def _time_full_throttle(self, threshold=0.90):
        apex_indx = find_closest_value(self.distance, self.apex_dist)
        for i in range(apex_indx, len(self.throttle)):
            if self.throttle[i] >= threshold:
                return self.time[i]
        return None

    def _combined_g(self):
        """Calcola quanto Grip il polota chiede alle gomme"""
        ax = np.array(self.acc_x)
        ay = np.array(self.acc_y)
        return np.sqrt(ax**2 + ay**2) / 9.81 #normalizzato in G
        
    #########################################################
    #                   Metrice semplici                    #
    #########################################################
        

    def brake_lead_time(self):
        """Tempo tra l'inizio della frenata e l'ingresso in curva"""
        t_brake = self._t_brake_on()
        if t_brake is None:
            return None
        return self._time_entry() - t_brake

    def time_to_throttle(self):
        """Tempo trascorso tra l'apex e la riapertura del gas (>20%)"""
        t_start = self._time_throttle_start()
        if t_start is None:
            return None
        return t_start - self._time_apex()

    def time_to_full_throttle(self):
        """Tempo trascorso tra l'apex e il raggiungimento del pieno gas (>=90%)"""
        t_full = self._time_full_throttle()
        if t_full is None:
            return None
        return t_full - self._time_apex()
    
    def peak_combined_g(self):
        return np.percentile(self._combined_g(), 99)

    def percentile_95_combined_g(self):
        return np.percentile(self._combined_g(), 95)

    def norm_combined_g(self):
        """Mi dice in percentuale quanto il pilota è stato vicino al picco di grip richiesto in quella curva """
        return self.percentile_95_combined_g()/self.peak_combined_g()



    #########################################################
    #                Metrice più complesse                  #
    #########################################################

<<<<<<< HEAD
=======

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
        # MAX_DAX: es. 1.0 → cambi di 6 g/s considerati "molto nervosi"
        MAX_DAX  = 6.0    # [g] per secondo

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


    def grip_usage(self, max_g: float = 6.0) -> float:
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
        Classifica lo stile di guida in una curva specifica.
        
        Stili disponibili:
        - aggressivo: alta aggressività, alto uso grip, trail braking marcato
        - medio: bilanciato tra prestazione e gestione
        - gestione: focus su stabilità, fluidità, conservazione energia
        
        La classificazione tiene conto di:
        1. Attacco (aggressività, grip usage, trail braking)
        2. Pulizia tecnica (stabilità, fluidità)
        3. Gestione energia (rapporto energia persa/immessa)
        4. Efficienza (quanto bene accelera in uscita)
        """
        
        # ============ RACCOLTA METRICHE ============
        stability      = curve.vehicle_stability()      # [0,1] → 1=stabile
        aggressiveness = curve.aggressiveness()         # [0,1] → 1=aggressivo
        fluidity       = curve.fluidity()               # [0,1] → 1=fluido
        efficiency     = curve.efficiency()             # variazione v²/dt
        energy_in      = curve.energy_input()           # energia motore
        energy_loss    = curve.energy_lost_brake()      # energia persa in frenata
        tbi            = curve.trail_braking_index()    # [0,1] → trail braking
        gui_raw            = curve.grip_usage()             # [0,1] → uso grip medio
        
        # Protezione divisione per zero
        eps = 1e-6
        
        # ============ INDICATORI COMPOSITI ============
        
        GUI_REF_MIN = 0.10
        GUI_REF_MAX = 0.65

        if gui_raw <= GUI_REF_MIN:
            gui = 0.0
        elif gui_raw >= GUI_REF_MAX:
            gui = 1.0
        else:
            gui = (gui_raw - GUI_REF_MIN) / (GUI_REF_MAX - GUI_REF_MIN)
        
        total_energy = energy_in + energy_loss + eps
        brake_ratio = energy_loss / total_energy
        brake_ratio = max(0.0, min(1.0, brake_ratio))
        

        cleanliness = 0.6 * stability + 0.4 * fluidity
        cleanliness = max(0.0, min(1.0, cleanliness))
        
    
        EFFICIENCY_REF = 50.0  # punto di riferimento
        efficiency_norm = 1.0 / (1.0 + math.exp(-efficiency / EFFICIENCY_REF))
        
        attack_primary = (
            0.40 * aggressiveness +
            0.45 * gui +
            0.30 * tbi -
            0.15 * brake_ratio  # segno negativo: troppo freno abbassa l'attacco
        )
        attack_primary = max(0.0, min(1.0, attack_primary))
        
    
        # Un pilota aggressivo MA efficiente e pulito è comunque aggressivo
        # Un pilota aggressivo MA inefficiente potrebbe essere solo scomposto
        attack_secondary = (
            0.50 * attack_primary +
            0.30 * efficiency_norm +
            0.25 * cleanliness
        )
        attack_secondary = max(0.0, min(1.0, attack_secondary))
        
       
        # Bilanciamo attacco primario (60%) con quello secondario (40%)
        # Questo permette di distinguere aggressività costruttiva da guida scomposta
        attack_score = 0.60 * attack_primary + 0.40 * attack_secondary
        
        # ============ FATTORI CORRETTIVI ============
        
        # Penalità per instabilità eccessiva
        # Se stability < 0.4 → probabile guida scomposta, non aggressiva
        if stability < 0.4:
            instability_penalty = (0.4 - stability) * 0.5  # max -0.2
            attack_score -= instability_penalty
        
        # Bonus per trail braking elevato CON buona stabilità
        # Trail braking efficace richiede controllo
        if tbi > 0.6 and stability > 0.6:
            tbi_bonus = (tbi - 0.6) * 0.15  # max +0.06
            attack_score += tbi_bonus
        
        # Bonus per alto grip usage CON buona efficienza
        # Usare tanto grip E accelerare bene = pilota veloce
        if gui > 0.55 and efficiency_norm > 0.6:
            grip_bonus = (gui - 0.55) * 0.2  
            attack_score += grip_bonus
        
        # Clamp finale
        attack_score = max(0.0, min(1.0, attack_score))
        
        # ============ SOGLIE DI CLASSIFICAZIONE ============
        
        # Soglie ottimizzate per riflettere stili reali:
        # - Qualifica: attack_score tipicamente 0.55-0.75 → aggressivo
        # - Gara push: attack_score 0.40-0.55 → medio
        # - Gara gestione: attack_score 0.20-0.40 → gestione
        
        ATTACK_AGGRESSIVE = 0.50   # >= 0.50 → aggressivo
        ATTACK_MANAGEMENT = 0.35   # <= 0.32 → gestione
        
        # ============ CLASSIFICAZIONE FINALE ============
        
        if attack_score >= ATTACK_AGGRESSIVE:
            # Ulteriore validazione per "aggressivo"
            # Deve avere almeno due di: alto grip, alto TBI, bassa brake_ratio
            aggressive_confirmations = sum([
                gui > 0.45,
                tbi > 0.45,
                brake_ratio < 0.45
            ])
            
            if aggressive_confirmations >= 2:
                return "aggressivo"
            else:
                # Falso positivo: probabilmente medio
                return "medio"
        
        elif attack_score <= ATTACK_MANAGEMENT:
            # Ulteriore validazione per "gestione"
            # Deve avere: bassa aggressività E (alta brake_ratio O basso grip)
            if aggressiveness < 0.35 and (brake_ratio > 0.45 or gui < 0.40):
                return "gestione"
            else:
                # Falso positivo: probabilmente medio
                return "medio"
        
        else:
            # Zona media: bilanciato
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
        sc = plt.scatter(x, y, c=spd, cmap='viridis')
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
        Nel titolo mostra il valore VSI aggregato [0..1].
        """
        kappa = np.array(self.curvature_profile(), dtype=float)
        
        if not self.speed or not self.acc_y or len(kappa) == 0:
            print("[plot_vehicle_stability] Dati insufficienti.")
            return

        v = np.asarray(self.speed, float)
        ay = np.asarray(self.acc_y, float)

        n = min(len(kappa), len(v), len(ay))
        if n < 5:
            print("[plot_vehicle_stability] Troppi pochi punti.")
            return

        kappa = kappa[:n]
        v = v[:n]
        ay = ay[:n]

        # Asse X
        if self.distance and len(self.distance) >= n:
            x_axis = np.asarray(self.distance[:n], float)
            x_label = "Distanza [m]"
        else:
            x_axis = np.arange(n)
            x_label = "Indice campione"

        # Filtro punti validi
        mask = (np.abs(kappa) > 1e-6) & (v > 1.0)
        
        if not np.any(mask):
            print("[plot_vehicle_stability] Nessun punto valido per la curva.")
            return

        x_eff = x_axis[mask]
        kappa_eff = kappa[mask]
        v_eff = v[mask]
        ay_eff = ay[mask]

        # Accelerazione laterale attesa
        ay_exp = v_eff * v_eff * kappa_eff

        # Differenza
        delta_ay = np.abs(ay_eff - ay_exp)
        mean_delta = float(np.mean(delta_ay))

        # USA IL METODO ESISTENTE
        vsi = self.vehicle_stability()

        # --- PLOT ---
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))

        # 1) ay misurata vs attesa
        ax1.plot(x_eff, ay_eff, label="ay misurata", linewidth=2)
        ax1.plot(x_eff, ay_exp, linestyle="--", label="ay attesa (v²·κ)", linewidth=2)
        ax1.set_ylabel("ay [m/s²]")
        ax1.set_title(f"Vehicle Stability Index (VSI) = {vsi:.3f}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2) |delta_ay|
        ax2.plot(x_eff, delta_ay, label="|ay - ay_exp|", linewidth=2)
        ax2.axhline(mean_delta, linestyle="--", color='red',
                    label=f"media |Δay| = {mean_delta:.3f}", linewidth=2)
        ax2.set_xlabel(x_label)
        ax2.set_ylabel("|Δay| [m/s²]")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


    def plot_curvature(self) -> None:
        """
        Profilo di curvatura κ lungo la distanza.
        Nel titolo mostra curvature_mean().
        """
        kappa = np.array(self.curvature_profile(), dtype=float)
        dist = np.array(self.distance, dtype=float)

        if len(kappa) != len(dist) or len(kappa) == 0:
            print("[plot_curvature] Dati insufficienti o mismatch dimensioni.")
            return

        # USA IL METODO ESISTENTE
        kappa_mean = self.curvature_mean()

        plt.figure(figsize=(10, 5))
        plt.plot(dist, kappa, linewidth=2)
        plt.xlabel("Distanza [m]")
        plt.ylabel("Curvatura κ [1/m]")
        plt.title(
            f"Corner {self.corner_id} – Profilo di curvatura\n"
            f"curvature_mean = {kappa_mean:.4f} [1/m]"
        )
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


    def plot_efficiency_profile(self, fraction: float = 0.10, use_time: bool = False) -> None:
        """
        Profilo di efficienza accelerativa:
        mostra la velocità lungo la curva e i segmenti usati per v_in e v_out.
        """
        if not self.speed or not self.distance or not self.time:
            print("[plot_efficiency_profile] Dati insufficienti.")
            return

        spd = np.asarray(self.speed, dtype=float)
        n = len(spd)
        amount = max(1, int(n * fraction))

        idx_in_end = amount
        idx_out_start = n - amount

        # USA I METODI ESISTENTI
        v_in = self.entry_speed_average()
        v_out = self.exit_speed_average()
        eff = self.efficiency()

        x = np.asarray(self.time if use_time else self.distance, dtype=float)
        x_label = "Tempo [s]" if use_time else "Distanza [m]"

        plt.figure(figsize=(10, 5))
        plt.plot(x, spd, label="Speed [m/s]", linewidth=2.5, color='blue')

        # Evidenzia zone di calcolo
        plt.axvspan(x[0], x[idx_in_end-1], alpha=0.2, color='green', label="zona v_in")
        plt.axvspan(x[idx_out_start], x[-1], alpha=0.2, color='orange', label="zona v_out")

        # Linee per v_in e v_out
        plt.axhline(v_in, linestyle="--", color='green', linewidth=2, label=f"v_in = {v_in:.1f} m/s")
        plt.axhline(v_out, linestyle="--", color='orange', linewidth=2, label=f"v_out = {v_out:.1f} m/s")

        plt.xlabel(x_label)
        plt.ylabel("Velocità [m/s]")
        plt.title(
            f"Corner {self.corner_id} – Profilo efficienza accelerativa\n"
            f"efficiency = (v_out² - v_in²)/dt = {eff:.3f} [m²/s³]"
        )
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


    def plot_energy_input_profile(self, use_time: bool = False) -> None:
        """
        Profilo di energy_input:
        mostra throttle*speed lungo la curva.
        L'area corrisponde alla metrica energy_input().
        """
        thr = np.asarray(self._normalize_pedal(self.throttle), dtype=float)
        spd = np.asarray(self.speed, dtype=float)
        t = np.asarray(self.time, dtype=float)
        dist = np.asarray(self.distance, dtype=float)

        if len(t) < 2 or len(thr) != len(t) or len(spd) != len(t):
            print("[plot_energy_input_profile] Dati insufficienti.")
            return

        dt = np.diff(t)
        if np.any(dt <= 0):
            print("[plot_energy_input_profile] dt non valido.")
            return

        # Profilo istantaneo
        power_like = thr[1:] * spd[1:]
        x_mid_time = 0.5 * (t[1:] + t[:-1])
        x_mid_dist = 0.5 * (dist[1:] + dist[:-1])
        x = x_mid_time if use_time else x_mid_dist
        x_label = "Tempo [s]" if use_time else "Distanza [m]"

        # USA IL METODO ESISTENTE
        E_in = self.energy_input()

        plt.figure(figsize=(10, 5))
        plt.plot(x, power_like, label="thr × speed", linewidth=2, color='green')
        plt.fill_between(x, 0.0, power_like, alpha=0.3, color='green')

        plt.xlabel(x_label)
        plt.ylabel("thr × speed [unità arbitrarie]")
        plt.title(
            f"Corner {self.corner_id} – Profilo energy_input\n"
            f"∫ thr·speed dt = {E_in:.3f} [u.a.]"
        )
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


    def plot_energy_lost_brake_profile(self, use_time: bool = False) -> None:
        """
        Profilo di energy_lost_brake:
        mostra brake × decelerazione lungo la curva.
        L'area totale è la metrica energy_lost_brake().
        """
        ax = np.asarray(self.acc_x, dtype=float)
        brk = np.asarray(self.brake, dtype=float)
        t = np.asarray(self.time, dtype=float)
        dist = np.asarray(self.distance, dtype=float)

        if len(t) < 2 or len(brk) != len(t) or len(ax) != len(t):
            print("[plot_energy_lost_brake_profile] Dati insufficienti.")
            return

        dt = np.diff(t)
        if np.any(dt <= 0):
            print("[plot_energy_lost_brake_profile] dt non valido.")
            return

        dec = np.maximum(-ax, 0.0)
        loss_inst = brk[1:] * dec[1:]
        x_mid_time = 0.5 * (t[1:] + t[:-1])
        x_mid_dist = 0.5 * (dist[1:] + dist[:-1])
        x = x_mid_time if use_time else x_mid_dist
        x_label = "Tempo [s]" if use_time else "Distanza [m]"

        # USA IL METODO ESISTENTE
        E_loss = self.energy_lost_brake()

        plt.figure(figsize=(10, 5))
        plt.plot(x, loss_inst, label="brake × dec", linewidth=2, color='red')
        plt.fill_between(x, 0.0, loss_inst, alpha=0.3, color='red')

        plt.xlabel(x_label)
        plt.ylabel("brake × dec [unità arbitrarie]")
        plt.title(
            f"Corner {self.corner_id} – Profilo energy_lost_brake\n"
            f"∫ brake·dec dt = {E_loss:.3f} [u.a.]"
        )
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


    def plot_aggressiveness_profile(self, use_time: bool = False) -> None:
        """
        Profilo di aggressività:
        - throttle normalizzato
        - derivata d(throttle)/dt
        Nel titolo: metrica aggressiveness().
        """
        if not self.time or not self.throttle or not self.acc_x:
            print("[plot_aggressiveness_profile] Dati insufficienti.")
            return

        t = np.asarray(self.time, dtype=float)
        dist = np.asarray(self.distance, dtype=float)
        thr = np.asarray(self._normalize_pedal(self.throttle), dtype=float)
        ax = np.asarray(self.acc_x, dtype=float)

        if len(t) < 2 or len(thr) != len(t) or len(ax) != len(t):
            print("[plot_aggressiveness_profile] Dati insufficienti (dimensioni).")
            return

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
        ax_min = float(np.min(ax))
        dec_peak = max(-ax_min, 0.0)
        
        # USA IL METODO ESISTENTE
        agg_val = self.aggressiveness()

        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Throttle
        x_full = t if use_time else dist
        ax1.plot(x_full, thr, label="Throttle (norm)", linewidth=2, color='blue')
        ax1.set_xlabel(x_label)
        ax1.set_ylabel("Throttle (0–1)", color="blue")
        ax1.tick_params(axis='y', labelcolor='blue')

        # Derivata throttle
        ax2 = ax1.twinx()
        ax2.plot(x, dthr_dt, linestyle="--", label="d(thr)/dt", linewidth=2, color='orange')
        ax2.set_ylabel("d(throttle)/dt [1/s]", color="orange")
        ax2.tick_params(axis='y', labelcolor='orange')

        # Legende
        l1, lab1 = ax1.get_legend_handles_labels()
        l2, lab2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, lab1 + lab2, loc="best")

        plt.title(
            f"Corner {self.corner_id} – Profilo aggressiveness\n"
            f"dec_peak = {dec_peak:.2f} m/s², max|dthr/dt| = {max_dthr:.2f} 1/s, "
            f"aggressiveness = {agg_val:.3f}"
        )
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


    def plot_fluidity_profile(self) -> None:
        """
        Visualizza la metrica di fluidità mostrando:
        - Le variazioni di throttle e acc_x (quanto sono "nervose")
        - Le derivate d(throttle)/dt e d(acc_x)/dt che misurano la dolcezza
        
        Più le derivate sono piccole → guida fluida
        Picchi nelle derivate → guida nervosa/a scatti
        """
        if not self.time or len(self.time) < 3:
            print("[plot_fluidity_profile] Dati insufficienti.")
            return

        t = np.asarray(self.time, dtype=float)
        thr = np.asarray(self._normalize_pedal(self.throttle), dtype=float)
        ax = np.asarray(self.acc_x, dtype=float) / 9.81  # in g

        n = min(len(t), len(thr), len(ax))
        if n < 3:
            print("[plot_fluidity_profile] Dati insufficienti.")
            return

        t = t[:n]
        thr = thr[:n]
        ax = ax[:n]

        # Calcola le derivate (variazioni nel tempo)
        dt = np.diff(t)
        valid = dt > 1e-6
        
        if not np.any(valid):
            print("[plot_fluidity_profile] dt non valido.")
            return
        
        dt_valid = dt[valid]
        dthr_dt = np.diff(thr)[valid] / dt_valid
        dax_dt = np.diff(ax)[valid] / dt_valid
        t_mid = 0.5 * (t[1:] + t[:-1])
        t_mid = t_mid[valid]

        # USA IL METODO ESISTENTE
        flu = self.fluidity()

        # Interpretazione del valore
        if flu >= 0.75:
            interpretation = "molto fluida"
        elif flu >= 0.50:
            interpretation = "abbastanza fluida"
        elif flu >= 0.30:
            interpretation = "un po' nervosa"
        else:
            interpretation = "molto nervosa"

        # --- PLOT CON DUE PANNELLI ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # PANNELLO 1: Valori originali
        ax1_twin = ax1.twinx()
        
        line1 = ax1.plot(t, thr, label="Throttle", linewidth=2.5, color='blue', alpha=0.8)
        line2 = ax1_twin.plot(t, ax, label="acc_x [g]", linewidth=2.5, color='red', alpha=0.8)
        
        ax1.set_ylabel("Throttle (0-1)", color='blue', fontsize=11, fontweight='bold')
        ax1_twin.set_ylabel("acc_x [g]", color='red', fontsize=11, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1_twin.tick_params(axis='y', labelcolor='red')
        ax1.set_title(f"Corner {self.corner_id} – Andamento Throttle e Accelerazione", 
                    fontsize=12, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        # Legenda combinata
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', fontsize=10)

        # PANNELLO 2: Derivate (misura della nervosità)
        ax2_twin = ax2.twinx()
        
        line3 = ax2.plot(t_mid, dthr_dt, label="Δ Throttle/Δt", 
                        linewidth=2, color='dodgerblue', alpha=0.7)
        line4 = ax2_twin.plot(t_mid, dax_dt, label="Δ acc_x/Δt [g/s]", 
                            linewidth=2, color='orangered', alpha=0.7)
        
        # Linee a zero per riferimento
        ax2.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        ax2.set_xlabel("Tempo [s]", fontsize=11, fontweight='bold')
        ax2.set_ylabel("Δ Throttle/Δt [1/s]", color='dodgerblue', 
                    fontsize=11, fontweight='bold')
        ax2_twin.set_ylabel("Δ acc_x/Δt [g/s]", color='orangered', 
                            fontsize=11, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='dodgerblue')
        ax2_twin.tick_params(axis='y', labelcolor='orangered')
        ax2.set_title("Variazioni (picchi = nervosità, valori bassi = fluidità)", 
                    fontsize=11, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.3)
        
        # Legenda combinata
        lines2 = line3 + line4
        labels2 = [l.get_label() for l in lines2]
        ax2.legend(lines2, labels2, loc='upper left', fontsize=10)

        # Titolo generale con interpretazione
        fig.suptitle(
            f"Fluidity Index = {flu:.3f} → Guida {interpretation}\n"
            f"(0 = molto nervosa, 1 = molto fluida)",
            fontsize=14, fontweight='bold', y=0.995
        )
        
        plt.tight_layout()
        plt.show()


    def plot_trail_braking_profile(self, use_time: bool = False) -> None:
        """
        Profilo di Trail Braking:
        - brake (0/1)
        - |acc_y|
        Nel titolo: Trail Braking Index (TBI) in [0,1].
        """
        if not self.time or not self.acc_y or not self.brake:
            print("[plot_trail_braking_profile] Dati insufficienti.")
            return

        t = np.asarray(self.time, dtype=float)
        dist = np.asarray(self.distance, dtype=float) if self.distance else np.arange(len(t))
        ay = np.asarray(self.acc_y, dtype=float)
        brk = np.asarray(self.brake, dtype=float)

        n = min(len(t), len(ay), len(brk), len(dist))
        if n < 2:
            print("[plot_trail_braking_profile] Troppi pochi punti.")
            return

        t = t[:n]
        dist = dist[:n]
        ay = ay[:n]
        brk = brk[:n]

        x = t if use_time else dist
        x_label = "Tempo [s]" if use_time else "Distanza [m]"

        ay_abs = np.abs(ay)
        curve_mask = ay_abs > 0.5
        brake_in_curve = (brk > 0.5) & curve_mask

        # USA IL METODO ESISTENTE
        tbi = self.trail_braking_index()

        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax1.plot(x, ay_abs, label="|acc_y| [m/s²]", linewidth=2, color='blue')
        ax1.fill_between(x, 0, ay_abs, where=brake_in_curve, alpha=0.4,
                        color='red', label="frenata in appoggio")

        ax2 = ax1.twinx()
        ax2.step(x, brk, where="post", linestyle="--", label="brake (0/1)", 
                linewidth=2, color='darkred')
        ax2.set_ylabel("Brake", color='darkred')
        ax2.tick_params(axis='y', labelcolor='darkred')

        ax1.set_xlabel(x_label)
        ax1.set_ylabel("|acc_y| [m/s²]", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_title(
            f"Corner {self.corner_id} – Trail Braking Profile\n"
            f"TBI = {tbi:.3f}"
        )

        l1, lab1 = ax1.get_legend_handles_labels()
        l2, lab2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, lab1 + lab2, loc="best")

        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


    def plot_grip_usage_profile(self, max_g: float = 6.0, use_time: bool = False) -> None:
        """
        Profilo di Grip Usage:
        - g_tot = sqrt((acc_x/g)^2 + (acc_y/g)^2)
        Nel titolo: Grip Usage Index (GUI) = mean(g_tot) / max_g.
        """
        if not self.acc_x or not self.acc_y or not self.time:
            print("[plot_grip_usage_profile] Dati insufficienti.")
            return

        ax = np.asarray(self.acc_x, dtype=float) / 9.81
        ay = np.asarray(self.acc_y, dtype=float) / 9.81
        t = np.asarray(self.time, dtype=float)
        dist = np.asarray(self.distance, dtype=float) if self.distance else np.arange(len(t))

        n = min(len(ax), len(ay), len(t), len(dist))
        if n == 0:
            print("[plot_grip_usage_profile] Troppi pochi punti.")
            return

        ax = ax[:n]
        ay = ay[:n]
        t = t[:n]
        dist = dist[:n]

        g_tot = np.sqrt(ax * ax + ay * ay)
        mean_g = float(np.mean(g_tot))
        
        # USA IL METODO ESISTENTE
        gui = self.grip_usage(max_g=max_g)

        x = t if use_time else dist
        x_label = "Tempo [s]" if use_time else "Distanza [m]"

        plt.figure(figsize=(10, 5))
        plt.plot(x, g_tot, label="g_tot (G combinato)", linewidth=2, color='purple')
        plt.axhline(mean_g, linestyle="--", color='blue', linewidth=2,
                    label=f"mean(g_tot) = {mean_g:.2f} G")
        plt.axhline(max_g, linestyle=":", color='red', linewidth=2,
                    label=f"max_g = {max_g:.2f} G")

        plt.xlabel(x_label)
        plt.ylabel("G combinato [g]")
        plt.title(
            f"Corner {self.corner_id} – Grip Usage Profile\n"
            f"GUI = {gui:.3f}"
        )
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
>>>>>>> bb2f3673385c3c01a5e8f7fa77fcc8eb3cab3334
