import math
from typing import List
import numpy as np
import matplotlib.pyplot as plt

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

    # --- Tempo & distanza ---
    def time_in_curve(self) -> float:
        return self.time[-1] - self.time[0] 

    def distance_in_curve(self) -> float:
        return self.distance[-1] - self.distance[0] 

    def speed_average(self) -> float:
        return sum(self.speed) / len(self.speed) 

    def entry_speed_average(self, fraction: float = 0.10) -> float:
        amount = max(1, int(len(self.speed) * fraction))
        entry_speeds = self.speed[:amount]
        return sum(entry_speeds) / len(entry_speeds) 

    def exit_speed_average(self, fraction: float = 0.10) -> float:
        amount = max(1, int(len(self.speed) * fraction))
        exit_speeds = self.speed[-amount:]
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


    # --- Traiettoria: curvatura a 3 punti ---
    def curvature_profile(self) -> List[float]:
        """
        Restituisce una lista kappa[k] (stessa lunghezza di x/y)
        calcolando il raggio basandosi su 3 punti:
        P1 = (x[k-1], y[k-1]), P2 = (x[k], y[k]), P3 = (x[k+1], y[k+1])
        """
        n = len(self.x)
        if n < 3:
            return [0.0] * n

        kappa = [0.0] * n  

        for k in range(1, n - 1):
            x1, y1 = self.x[k - 1], self.y[k - 1]
            x2, y2 = self.x[k],     self.y[k]
            x3, y3 = self.x[k + 1], self.y[k + 1]

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
        Faccio una media integrale essendo che le curveture non sono sempre equi diistanti 
        ogni curvatoura e oresa su un intervallo di spazio diverso
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

        ax_min = min(self.acc_x)         
        dec_peak = max(-ax_min, 0.0)  #quanto forte freni (fisicamente: G di decelerazione)

    
        dthr_dt = np.diff(thr) / np.diff(self.time) #formula derivata

        max_dthr = np.max(np.abs(dthr_dt)) if len(dthr_dt) > 0 else 0.0 #quanto sei brusco nel cambiare gas (derivata del throttle)

        return dec_peak * (1.0 + max_dthr)

    def fluidity(self) -> float:
        """
        Fluidità dei comandi:
        più throttle, brake e acc_x sono stabili (bassa varianza),
        più la fluidità è alta.

        brake è 0/1 -> lo prendiamo così com'è.
        """
        if not self.throttle or not self.brake or not self.acc_x:
            return 0.0

        thr = np.asarray(self._normalize_pedal(self.throttle), dtype=float)
        brk = np.asarray(self.brake, dtype=float)      
        ax  = np.asarray(self.acc_x, dtype=float)

    
        n = min(len(thr), len(brk), len(ax))
        if n == 0:
            return 0.0

        thr = thr[:n]
        brk = brk[:n]
        ax  = ax[:n]

        var_thr = float(np.var(thr))
        var_brk = float(np.var(brk))
        var_ax  = float(np.var(ax))

        mean_var = (var_thr + var_brk + var_ax) / 3.0
        eps = 1e-6
        return 1.0 / (mean_var + eps)                   # Più i tre segnali oscillano, più la varianza cresce → 1/(var) scende → fluidità bassa




    # def efficiency(self, fraction: float = 0.10) -> float:
    #     """
    #     Efficienza accelerativa:
    #     (v_out^2 - v_in^2) / dt_curva

    #     v_in e v_out stimati come media sugli ultimi/primissimi 'fraction' punti.
    #     """
    #     dt = self.time_in_curve()
    #     if dt <= 0 or not self.speed:
    #         return 0.0

    #     spd = np.asarray(self.speed, dtype=float)
    #     n = len(spd)
    #     amount = max(1, int(n * fraction))

    #     v_in  = float(spd[:amount].mean())
    #     v_out = float(spd[-amount:].mean())

    #     return (v_out**2 - v_in**2) / dt


    # def stability(self) -> float:
    #     """
    #     Stabilità laterale:
    #     std(|acc_y|) / max(|acc_y|)

    #     Più il rapporto è piccolo → curva "pulita" e stabile.
    #     """
    #     if not self.acc_y:
    #         return 0.0

    #     ay = np.asarray(self.acc_y, dtype=float)
    #     ay_abs = np.abs(ay)

    #     ay_max = float(ay_abs.max())
    #     if ay_max <= 0:
    #         return 0.0

    #     std_ay = float(ay_abs.std())   # np.std

    #     return std_ay / ay_max

    def plot_controls(self, use_time: bool = False) -> None:
        """
        Grafico combinato:
        - Velocità
        - Throttle normalizzato
        - Brake

        Se use_time = False -> asse X = distanza
        Se use_time = True  -> asse X = tempo
        """
        if not self.time or not self.distance or not self.speed:
            print("[plot_controls] Dati insufficienti.")
            return

        x_label = "Tempo [s]" if use_time else "Distanza [m]"
        x = np.array(self.time if use_time else self.distance, dtype=float)

        spd = np.array(self.speed, dtype=float)
        thr = np.array(self._normalize_pedal(self.throttle), dtype=float)
        brk = np.array(self.brake, dtype=float)

        fig, ax1 = plt.subplots()

        # Asse principale: velocità
        ax1.plot(x, spd, label="Speed", linewidth=2)
        ax1.set_xlabel(x_label)
        ax1.set_ylabel("Velocità", color="black")

        # Asse secondario: pedali
        ax2 = ax1.twinx()
        ax2.plot(x, thr, linestyle="--", label="Throttle (norm)")
        ax2.plot(x, brk, linestyle=":", label="Brake")
        ax2.set_ylabel("Pedali (0–1)", color="black")

        # Legenda combinata
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        plt.title(f"Corner {self.corner_id} – Speed / Throttle / Brake")
        plt.tight_layout()
        plt.show()

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

    def plot_gg_diagram(self) -> None:
        """
        G-G diagram locale alla curva: acc_x vs acc_y.
        Serve a vedere come usi il grip (longitudinale vs laterale).
        """
        if not self.acc_x or not self.acc_y:
            print("[plot_gg_diagram] Dati insufficienti.")
            return

        ax = np.array(self.acc_x, dtype=float)
        ay = np.array(self.acc_y, dtype=float)

        plt.figure()
        plt.scatter(ax, ay, s=8)
        plt.axhline(0.0, linewidth=0.5)
        plt.axvline(0.0, linewidth=0.5)
        plt.xlabel("acc_x [m/s²] (longitudinale)")
        plt.ylabel("acc_y [m/s²] (laterale)")
        plt.title(f"Corner {self.corner_id} – G-G diagram")
        plt.axis("equal")
        plt.tight_layout()
        plt.show()