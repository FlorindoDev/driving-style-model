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
        return sum(entry_speeds) / len(entry_speeds) 

    def exit_speed_average(self) -> float:
        apex_pilot_idx = find_closest_value(self.distance, self.apex_dist)
        exit_speeds = self.speed[apex_pilot_idx:]
        return sum(exit_speeds) / len(exit_speeds) 
    
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


    def curvature_profile(self) -> List[float]:
        """
        Restituisce una lista kappa[k] (stessa lunghezza di x/y)
        calcolando il raggio di un cerchio passante su 3 punti:
        P1 = (x[k-1], y[k-1]), P2 = (x[k], y[k]), P3 = (x[k+1], y[k+1])
        """
        n = len(self.x)
        if n < 3:
            return [0.0] * n

        kappa = [0.0] * n  

        # 3 poiché punti troppo vicini una piccola varianzione puo creare dei spike nel grafico
        for k in range(5, n - 5):
            x1, y1 = self.x[k - 5], self.y[k - 5]
            x2, y2 = self.x[k],     self.y[k]
            x3, y3 = self.x[k + 5], self.y[k + 5]

            a = math.hypot(x3 - x2, y3 - y2)
            b = math.hypot(x3 - x1, y3 - y1)
            c = math.hypot(x2 - x1, y2 - y1)

            area = 0.5 * abs(
                (x2 - x1) * (y3 - y1) -
                (y2 - y1) * (x3 - x1)
            )

            # Se i punti sono quasi allineati o troppo vicini, curvatura = 0
            if area < 1e-9 or a == 0.0 or b == 0.0 or c == 0.0:
                kappa[k] = 0.0
                continue

            # formula Raggio del cerchio circoscritto fra tre punti
            R = (a * b * c) / (4.0 * area)

            # Curvatura = 1 / R 
            # Per R grande e più la Curvatura sarà piccola 
            # più R è piccolo più stretta sarà la curva 
            kappa[k] = 1.0 / R

  
        return kappa

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
        Misura l'aggressività nella curva combinando:
        - picco di decelerazione in frenata (acc_x_min)
        - bruschezza del gas (max derivata del throttle).

        brake è binario (0/1), quindi l'intensità della frenata
        la leggiamo da acc_x, non dal valore del pedale.
        """

        thr = self._normalize_pedal(self.throttle)

        if not self.time or not thr:
            return 0.0

        t  = np.asarray(self.time, float)
        ax = np.asarray(self.acc_x, float)

        dec_peak = max(-np.min(ax), 0.0)   

   
        dthr_dt = np.diff(thr) / np.diff(t)
        dt      = np.diff(t)

        if len(dthr_dt) == 0:
            return 0.0

        abs_d = np.abs(dthr_dt)

       
        max_d = float(abs_d.max())
        if max_d < 1e-6:
            mean_abs_d = 0.0
        else:
            # prendo i punti con derivata almeno al 20% del massimo di quella curva
            DERIV_REL = 0.2  
            thr_val = DERIV_REL * max_d

            abs_d_filtered = abs_d[abs_d >= thr_val]
            dt_filtered    = dt[abs_d >= thr_val]

            # se per qualche motivo rimane vuoto, fallback alla media normale
            if len(abs_d_filtered) == 0:
                mean_abs_d = float(np.sum(abs_d * dt) / (t[-1] - t[0]))
            else:
                mean_abs_d = float(np.sum(abs_d_filtered * dt_filtered) / np.sum(dt_filtered))

   
        DEC_MAX = 60.0
        dec_norm = min(dec_peak / DEC_MAX, 1.0)

        # da tarare con dati reali questo perchè il tempo di risposta biologiva massimo di 0,2 quindi 1/0,2 = 5 
        # quindi la derivata del intervallo massimo dell accleatore fratto il tempo minimo di reazione
        MEAN_D_MAX = 2
        gas_norm = min(mean_abs_d / MEAN_D_MAX, 1.0)

        # Aggressività combinata o solo gas se non freni
        if dec_norm != 0:
            aggr_norm = dec_norm * gas_norm
        else:
            aggr_norm = gas_norm

        return aggr_norm


    def fluidity(self) -> float:
        """
        Fluidità dei comandi:
        misura quanto throttle e acc_x sono "smooth".
        Restituisce un punteggio tra ~0 e 1:
        - vicino a 1 -> molto fluido
        - vicino a 0 -> molto sporco
        """
        if not self.throttle or not self.acc_x:
            return 0.0

        thr = np.asarray(self._normalize_pedal(self.throttle), dtype=float)
        ax  = np.asarray(self.acc_x, dtype=float)

        n = min(len(thr), len(ax))
        if n < 2:
            return 0.0

        thr = thr[:n]
        ax  = ax[:n]

        # normalizza acc_x per non farlo dominare ([-1, 1] circa)
        ax_abs_max = float(np.max(np.abs(ax))) if np.max(np.abs(ax)) > 0 else 1.0
        ax_norm = ax / ax_abs_max

        var_thr = float(np.var(thr))
        var_ax  = float(np.var(ax_norm))

        mean_var = (var_thr + var_ax) / 2.0
       
        fluidity_score = 1.0 / (1.0 + mean_var)

        return fluidity_score



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


    def stability(self) -> float:
        """
        Stabilità laterale:
        std(|acc_y|) / max(|acc_y|)

        Misura quanto la forza laterale oscilla rispetto al suo livello massimo(normalizzata per max(|acc_y|)).
        La normalizzazione con max(|acc_y|) rende il valore comparabile tra curve
        di intensità diversa: a parità di oscillazioni, una curva dolce è meno
        stabile di una curva stretta. Valori bassi = curva fluida e stabile.
        """
        if not self.acc_y:
            return 0.0

        ay = np.asarray(self.acc_y, dtype=float)
        ay_abs = np.abs(ay)

        ay_max = float(ay_abs.max())
        if ay_max <= 0:
            return 0.0

        std_ay = float(ay_abs.std())   # np.std

        return std_ay / ay_max

    #########################################################
    #                 Calcolo Stile di guida                #
    #########################################################

    @staticmethod
    def classify_driver_style(curve: "Curve") -> str:
        stability       = curve.stability()
        aggressiveness  = curve.aggressiveness()
        fluidity        = curve.fluidity()           # supponiamo già "aggiustata"
        energy_in       = curve.energy_input()
        energy_loss     = curve.energy_lost_brake()
        efficiency      = curve.efficiency()

        # per evitare problemi di divisione per zero
        eps = 1e-6
        energy_in_safe = energy_in if abs(energy_in) > eps else eps
        den = energy_loss + energy_in_safe

        if den < eps:
            brake_ratio = 0.0
        else:
            brake_ratio = energy_loss / den  # quanta energia butti via in frenata, è una precentuale quanta energia perso sul totale che avevo
        

        # 1) OVERDRIVE: tanto instabile + molto aggressivo + efficienza negativa
        if stability > 0.38 and aggressiveness > 18 and efficiency < 0 and brake_ratio > 0.8:
            return "overdrive"

        # 2) SPORCO / INSTABILE: instabilità alta o fluidità molto bassa
        if stability > 0.30 or fluidity < 0.3:
            return "sporco / instabile"

        # 3) AGGRESSIVO: freni forte o butti via tanta energia in frenata
        if aggressiveness > 15 or brake_ratio > 0.6:
            return "aggressivo"

        # 4) FLUIDO: stabile, fluido, non troppo aggressivo
        if stability < 0.22 and fluidity > 0.6 and aggressiveness < 12:
            return "fluido"

        # 5) CONSERVATIVO: spinge poco e non è aggressivo
        if energy_in < 0.4 * energy_loss and aggressiveness < 8 and efficiency >= 0:
            return "conservativo"

        # 6) NEUTRO
        return "neutro"




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
        plt.colorbar(sc, label="Speed")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.title(f"Corner {self.corner_id} – Traiettoria colorata per velocità")
        plt.axis("equal")
        plt.tight_layout()
        plt.show()

    def plot_curvature(self) -> None:
        """
        Profilo di curvatura κ lungo la distanza.
        Usa curvature_profile() e self.distance.
        """
        kappa = np.array(self.curvature_profile(), dtype=float)
        dist = np.array(self.distance, dtype=float)

        if len(kappa) != len(dist) or len(kappa) == 0:
            print("[plot_curvature] Dati insufficienti o mismatch dimensioni.")
            return

        plt.figure()
        plt.plot(dist, kappa)
        plt.xlabel("Distanza [m]")
        plt.ylabel("Curvatura κ [1/m]")
        plt.title(f"Corner {self.corner_id} – Profilo di curvatura")
        plt.tight_layout()
        plt.show()


    def plot_stability_profile(self, use_time: bool = False) -> None:
        """
        Profilo di stabilità laterale:
        |acc_y| lungo la curva, con media e ±1 deviazione standard.

        Utile per vedere se la curva è "pulita" o se ci sono oscillazioni
        evidenti della forza laterale.
        """
        if not self.acc_y or not self.distance or not self.time:
            print("[plot_stability_profile] Dati insufficienti.")
            return

        ay = np.asarray(self.acc_y, dtype=float)
        ay_abs = np.abs(ay)

        mean_ay = float(ay_abs.mean())
        std_ay = float(ay_abs.std())

        x = np.asarray(self.time if use_time else self.distance, dtype=float)
        x_label = "Tempo [s]" if use_time else "Distanza [m]"

        stab = self.stability()

        plt.figure()
        plt.plot(x, ay_abs, label="|acc_y|", linewidth=1.5)
        plt.axhline(mean_ay, linestyle="--", label="media |acc_y|")
        plt.axhline(mean_ay + std_ay, linestyle=":", label="media + σ")
        plt.axhline(mean_ay - std_ay, linestyle=":", label="media - σ")

        plt.xlabel(x_label)
        plt.ylabel("|acc_y| [m/s²]")
        plt.title(
            f"Corner {self.corner_id} – Profilo stabilità laterale\n"
            f"stability = std(|acc_y|)/max(|acc_y|) = {stab:.3f}"
        )
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()


    def plot_efficiency_profile(self, fraction: float = 0.10, use_time: bool = False) -> None:
        """
        Profilo di efficienza accelerativa:
        mostra la velocità lungo la curva e i segmenti usati per stimare
        v_in e v_out nella metrica di efficienza.
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
        plt.plot(x, spd, label="Speed", linewidth=2)

        # evidenzia i segmenti di ingresso/uscita usati per la metrica
        plt.axvspan(x[0], x[idx_in_end-1], alpha=0.2, label="zona v_in")
        plt.axvspan(x[idx_out_start], x[-1], alpha=0.2, label="zona v_out")

        # linee orizzontali per v_in e v_out
        plt.axhline(v_in, linestyle="--", label=f"v_in ≈ {v_in:.1f}")
        plt.axhline(v_out, linestyle=":", label=f"v_out ≈ {v_out:.1f}")

        plt.xlabel(x_label)
        plt.ylabel("Velocità [unità]")
        plt.title(
            f"Corner {self.corner_id} – Profilo efficienza accelerativa\n"
            f"efficiency = (v_out² - v_in²)/dt = {eff:.3f}"
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
        thr = np.asarray(self._normalize_pedal(self.throttle), dtype=float)
        spd = np.asarray(self.speed, dtype=float)
        t   = np.asarray(self.time, dtype=float)
        dist = np.asarray(self.distance, dtype=float)

        if len(t) < 2 or len(thr) != len(t) or len(spd) != len(t):
            print("[plot_energy_input_profile] Dati insufficienti.")
            return

        dt = np.diff(t)
        if np.any(dt <= 0):
            print("[plot_energy_input_profile] dt non valido.")
            return

        # integrando locale: throttle * speed * dt
        power_like = thr[1:] * spd[1:]          # parte “istantanea”
        x_mid_time  = 0.5 * (t[1:] + t[:-1])
        x_mid_dist  = 0.5 * (dist[1:] + dist[:-1])
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
            f"∫ thr*speed dt ≈ {E_in:.3f}"
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
        ax  = np.asarray(self.acc_x, dtype=float)
        brk = np.asarray(self.brake, dtype=float)
        t   = np.asarray(self.time, dtype=float)
        dist = np.asarray(self.distance, dtype=float)

        if len(t) < 2 or len(brk) != len(t) or len(ax) != len(t):
            print("[plot_energy_lost_brake_profile] Dati insufficienti.")
            return

        dt = np.diff(t)
        if np.any(dt <= 0):
            print("[plot_energy_lost_brake_profile] dt non valido.")
            return

        dec = np.maximum(-ax, 0.0)             # solo decelerazione
        loss_inst = brk[1:] * dec[1:]          # parte “istantanea”
        x_mid_time  = 0.5 * (t[1:] + t[:-1])
        x_mid_dist  = 0.5 * (dist[1:] + dist[:-1])
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
            f"∫ brake * (-acc_x) dt ≈ {E_loss:.3f}"
        )
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()


    def plot_aggressiveness_profile(self, use_time: bool = False) -> None:
        """
        Profilo di aggressività:
        - throttle normalizzato
        - derivata d(throttle)/dt
        - evidenzia il picco di decelerazione e il max |dthr/dt|

        Nel titolo viene riportata la metrica aggressiveness().
        """
        if not self.time or not self.throttle or not self.acc_x:
            print("[plot_aggressiveness_profile] Dati insufficienti.")
            return

        t   = np.asarray(self.time, dtype=float)
        dist = np.asarray(self.distance, dtype=float)
        thr = np.asarray(self._normalize_pedal(self.throttle), dtype=float)
        ax  = np.asarray(self.acc_x, dtype=float)

        if len(t) < 2 or len(thr) != len(t) or len(ax) != len(t):
            print("[plot_aggressiveness_profile] Dati insufficienti (dimensioni).")
            return

        # derivata del gas
        dt = np.diff(t)
        if np.any(dt <= 0):
            print("[plot_aggressiveness_profile] dt non valido.")
            return
        dthr_dt = np.diff(thr) / dt
        x_mid_time  = 0.5 * (t[1:] + t[:-1])
        x_mid_dist  = 0.5 * (dist[1:] + dist[:-1])
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
            f"dec_peak = {dec_peak:.2f}, max|dthr/dt| = {max_dthr:.2f}, "
            f"aggressiveness = {agg_val:.3f}"
        )
        plt.tight_layout()
        plt.show()


    def plot_fluidity_profile(self, use_time: bool = False) -> None:
        """
        Profilo di fluidità:
        mostra throttle, brake e acc_x (scalato) sulla stessa X.
        Serve a vedere visivamente quanto i segnali sono “smooth” o seghettati.
        Nel titolo compare la metrica fluidity().
        """
        if not self.time or not self.throttle or not self.brake or not self.acc_x:
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
            f"fluidity = {flu:.3f} (1/var(thr,acc_x))"
        )
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()
