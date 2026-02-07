import math
from typing import List, Optional, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
import warnings


class Curve:
         
    def __init__(
        self,
        corner_id: int,
        current_corner_dist: float,
        lower_bound: float,
        upper_bound: float,
        compound: str,
        life: int,
        stint: int,
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
        z: List[float]
    ):
        self.corner_id = corner_id
        self.current_corner_dist = current_corner_dist
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.time = np.asarray(time)
        self.rpm = np.asarray(rpm)
        self.speed = np.asarray(speed)
        self.throttle = np.asarray(throttle)
        self.brake = np.asarray(brake)
        self.distance = np.asarray(distance)
        self.acc_x = np.asarray(acc_x)
        self.acc_y = np.asarray(acc_y)
        self.acc_z = np.asarray(acc_z)
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.z = np.asarray(z)
        self.compound = compound
        self.life = life
        self.stint = stint
        self.latent_variable = None
        self.num_cluster = None

    @classmethod
    def from_norm_data(
        cls,
        sample: np.ndarray,
        mask: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        latent_variable: Optional[List[float]] = None,
        num_cluster: Optional[int] = None,
    ):
        """
        sample: vettore 1D con layout:
          0: life
          1:51 speed
          51:101 rpm
          101:151 throttle
          151:201 brake
          201:251 acc_x
          251:301 acc_y
          301:351 acc_z
          352,353: compound one-hot (hard/medium) altrimenti soft
        mask: boolean mask compatibile (stesse slice)
        """
        compound = ""
        # compound
        if sample[352] != 0:
            compound = "HARD"
        elif sample[353] != 0:
            compound = "INTERMEDIATE"
        elif sample[354] != 0:
            compound = "WET"
        elif sample[355] != 0:
            compound = "MEDIUM"
        else:
            compound = "SOFT"

        bool_mask = mask.astype(bool)

        life = int(sample[0])  # se life non è mascherato, meglio così
        life = cls._denormalize_value(cls, life, mean[0], std[0]) 
        
        speed = cls._extract_and_denormalize(cls, 1, 51, sample, bool_mask, mean, std)
        rpm = cls._extract_and_denormalize(cls, 51, 101, sample, bool_mask, mean, std)
        throttle = cls._extract_and_denormalize(cls, 101, 151, sample, bool_mask, mean, std)
        brake = cls._extract_and_denormalize(cls, 151, 201, sample, bool_mask, mean, std)
        acc_x = cls._extract_and_denormalize(cls, 201, 251, sample, bool_mask, mean, std)
        acc_y = cls._extract_and_denormalize(cls, 251, 301, sample, bool_mask, mean, std)
        acc_z = cls._extract_and_denormalize(cls, 301, 351, sample, bool_mask, mean, std)

        # Create the instance
        instance = cls(
            corner_id=-1,
            current_corner_dist=-1,
            lower_bound=-1,
            upper_bound=-1,
            compound=compound,
            life=life,
            stint=-1,
            time=list(),
            rpm=rpm,
            speed=speed,
            throttle=throttle,
            brake=brake,
            distance=list(),
            acc_x=acc_x,
            acc_y=acc_y,
            acc_z=acc_z,
            x=list(),
            y=list(),
            z=list()
        )
        
        # Set optional attributes
        instance.latent_variable = latent_variable
        instance.num_cluster = num_cluster

        
        
        return instance


    #########################################################
    #                   Metodi privati                      #
    #########################################################

    def _denormalize_array(self, arr: np.ndarray, mean: float, std: float) -> np.ndarray:
        return (arr * std) + mean
    
    def _denormalize_value(self, value: float, mean: float, std: float) -> float:
        return (value * std) + mean
    
    def _extract_and_denormalize(
        self,
        start: int,
        end: int,
        sample: np.ndarray,
        bool_mask: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray
    ) -> list:
        """
        Estrae una porzione dell'array sample[start:end], applica la maschera e denormalizza.
        
        Args:
            start: indice di inizio (incluso)
            end: indice di fine (escluso)
            sample: array completo dei dati normalizzati
            bool_mask: maschera booleana per filtrare i valori validi
            mean: array delle medie per la denormalizzazione
            std: array delle deviazioni standard per la denormalizzazione
        
        Returns:
            Lista dei valori estratti, filtrati e denormalizzati
        """
        extracted = sample[start:end][bool_mask[start:end]].tolist()
        return self._denormalize_array(self, np.asarray(extracted), mean[start], std[start]).tolist()

    def print(self):
        print(f"[!] apex:{self.apex_dist}; start: {self.distance[0]} -> end: {self.distance[-1]}")

    

    #########################################################
    #                   Metriche semplici                    #
    #########################################################

    def lateral_g(self) -> np.ndarray:
        """G laterali (curva)"""
        return self.acc_y / 9.81

    def longitudinal_g(self) -> np.ndarray:
        """G longitudinali (positivo=accelerazione, negativo=frenata)"""
        return self.acc_x / 9.81

    def vertical_g(self) -> np.ndarray:
        """G verticali"""
        return self.acc_z / 9.81

    def total_g_xy(self) -> np.ndarray:
        """G totali sul piano XY (grip utilizzato)"""
        acc_total = np.sqrt(self.acc_x**2 + self.acc_y**2)
        return acc_total / 9.81

    def brake_intensity(self) -> np.ndarray:
        """Intensità frenata (0 quando non frena, alto quando frena forte)"""
        return self.brake * np.abs(np.minimum(self.acc_x, 0)) #confronta acc_x con zero e ritorna il più piccolo 

    def throttle_rate(self) -> np.ndarray:
        """Velocità di applicazione throttle (quanto velocemente apre/chiude)"""
        return np.gradient(self.throttle)

    def brake_time_percent(self) -> float:
        """Percentuale tempo in frenata"""
        return (np.sum(self.brake) / len(self.brake)) * 100

    def full_throttle_percent(self) -> float:
        """Percentuale tempo a gas spalancato (>95%)"""
        return (np.sum(self.throttle > 95) / len(self.throttle)) * 100

    def avg_speed(self) -> float:
        """Velocità media"""
        return np.mean(self.speed)

    def max_speed(self) -> float:
        """Velocità massima"""
        return np.max(self.speed)

    def tire_wear_total(self) -> float:
        """Degrado totale gomme durante il periodo analizzato"""
        if isinstance(self.life, (int, float, np.number)):
            return 0.0
        return self.life[0] - self.life[-1] if len(self.life) > 0 else 0

    #########################################################
    #                Metriche più complesse                 #
    #########################################################

    def grip_usage_percent(self, max_g: float = 6.5) -> np.ndarray:
        """
        Percentuale utilizzo cerchio di aderenza
        max_g: massimo G teorico raggiungibile (default 6.5)
        Ritorna: array con % utilizzo grip momento per momento
        """
        return (self.total_g_xy() / max_g) * 100

    def aggressivity_index(self) -> np.ndarray:
        """
        Indice aggressività relativo al grip disponibile.
        Stima quanto stai usando del grip residuo della gomma.
        """
        # Stima degradazione grip: perde circa 1.5-2% per giro
        DEGRADATION_PER_LAP = 0.02  # 2% perdita grip per giro
        
        # Grip residuo stimato (da 1.0 a ~0.4 per gomme molto vecchie)
        grip_remaining = max(1.0 - (self.life * DEGRADATION_PER_LAP), 0.4)
        
        # G massimi teorici (es. 6.5G) scalati per grip residuo
        MAX_G = 6.5
        available_g = MAX_G * grip_remaining
        
        # Quanto stai usando del grip disponibile
        acc_total = np.sqrt(self.acc_x**2 + self.acc_y**2)
        return (acc_total / available_g) * 100  # percentuale utilizzo

    def smoothness_index(self, window: int = 20) -> float:
        """
        Indice di fluidità guida
        Basso (< 0.3) = guida fluida, gestione
        Alto (> 0.5) = guida nervosa, spinta al limite
        """
        acc_x = np.asarray(self.acc_x)
        acc_y = np.asarray(self.acc_y)
        acc_total = np.sqrt(acc_x**2 + acc_y**2)
        
        if len(acc_total) < window:
            window = max(len(acc_total) // 2, 2)
        
        # Rolling standard deviation
        rolling_std = np.array([np.std(acc_total[max(0, i-window):i+1]) for i in range(len(acc_total))])
        mean_acc = np.mean(acc_total)
        
        return np.mean(rolling_std) / mean_acc if mean_acc > 0 else 0

    def corner_speed_index(self, lateral_g_threshold: float = 0.3) -> float:
        """
        Velocità media nelle curve (dove G laterali > threshold)
        Alto = entra veloce in curva (spingendo)
        """
        in_corner = np.abs(self.lateral_g()) > lateral_g_threshold
        if not np.any(in_corner):
            return 0.0
        return np.mean(self.speed[in_corner])

    def corner_aggression(self, lateral_g_threshold: float = 0.3) -> float:
        """
        Aggressività in curva = velocità × G laterali
        Più alto = più aggressivo in curva
        """
        lateral_g = self.lateral_g()
        in_corner = np.abs(lateral_g) > lateral_g_threshold
        if not np.any(in_corner):
            return 0.0
        
        avg_speed = np.mean(self.speed[in_corner])
        avg_lat_g = np.mean(np.abs(lateral_g[in_corner]))
        
        return avg_speed * avg_lat_g

    def tire_stress_score(self) -> float:
        """
        Score di stress sulle gomme
        Combina accelerazione totale, velocità e usura
        Gomme più vecchie = stress amplificato (più rischioso)
        """
        acc_total = np.sqrt(self.acc_x**2 + self.acc_y**2)
        # Moltiplicatore usura: da 1.0 (gomma nuova) a ~2.0 (30 giri)
        wear_multiplier = 1 + (self.life / 30)
        stress = acc_total * self.speed * wear_multiplier
        return np.mean(stress)

    def braking_aggression(self) -> float:
        """
        Aggressività in frenata - Versione migliorata
        Un pilota che spinge:
        - Frena TARDI (alta velocità all'inizio frenata)
        - Frena FORTE (alta decelerazione media)
        - Frena BREVE (rilascio rapido)
        """
        brake = np.asarray(self.brake)
        brake_mask = brake == 1
        if not np.any(brake_mask):
            return 0.0
        
        # 1. Decelerazione MEDIA durante frenata (non solo massima)
        avg_decel_g = np.mean(np.abs(self.longitudinal_g()[brake_mask]))
        
        # 2. Velocità all'ingresso in frenata (più alta = più aggressivo)
        speed_at_brake = np.mean(self.speed[brake_mask])
        speed_factor = speed_at_brake / 300  # normalizza su 300 km/h
        
        # 3. "Brevità" frenata: meno tempo = più aggressivo (inversione)
        brake_brevity = 1 - (self.brake_time_percent() / 100)
        
        # Score combinato (pesi da calibrare)
        score = (avg_decel_g * 20) * speed_factor * (0.5 + brake_brevity)
        
        return np.clip(score, 0, 100)       
            
  
    
    def pushing_score(self) -> float:
        """
        SCORE PRINCIPALE: indica se sta spingendo (0-100)
        
        < 30: GESTIONE - Pilota conservativo, gestisce gomme/macchina
        30-60: RITMO - Guida veloce ma controllata
        > 60: SPINTA - Sta spingendo al limite
        > 80: QUALIFICA - Massima spinta, un giro secco
        """
        # Grip usage assoluto (G totali vs max teorico)
        grip_score = np.clip(np.mean(self.grip_usage_percent()), 0, 100)
        
        # Aggressivity: quanto stai usando del grip DISPONIBILE (considera usura gomme)
        aggressivity_score = np.clip(np.mean(self.aggressivity_index()), 0, 100)
        
        # Smoothness: alto smoothness = guida nervosa = spinta alta
        # (smoothness_index alto significa variazioni frequenti nelle accelerazioni)
        smooth = self.smoothness_index()
        smoothness_score = np.clip(smooth * 100, 0, 100)
        
        # Throttle usage
        throttle_score = self.full_throttle_percent()
        
        # Corner aggression normalizzato
        corner_agg = self.corner_aggression()
        corner_score = np.clip(corner_agg * 0.5, 0, 100)
        
        # Braking aggression
        brake_agg = self.braking_aggression()
        brake_score = np.clip(brake_agg, 0, 100)
        
        # Fattore velocità: penalizza giri lenti (outlap, inlap, gestione estrema)
        # Reference speed per curve F1: ~120 km/h media in curva durante spinta
        avg_spd = self.avg_speed()
        REFERENCE_SPEED = 120.0
        speed_factor = np.clip(avg_spd / REFERENCE_SPEED, 0.0, 1.0)
        
        # Media ponderata - pesi ribilanciati:
        # - grip e aggressivity pesano di più (sono la misura diretta dei G)
        # - corner_score peso aumentato (riflette velocità in curva)
        # - smoothness ridotto (meno affidabile come indicatore)
        raw_score = (
            grip_score * 0.25 +          # G assoluti - peso principale
            aggressivity_score * 0.25 +  # G relativi al grip disponibile
            throttle_score * 0.10 +      # Uso gas
            corner_score * 0.20 +        # Aggressività in curva
            brake_score * 0.10 +         # Aggressività frenata
            smoothness_score * 0.10      # Variabilità guida
        )
        
        # Applica penalità velocità: score finale scalato per speed_factor
        total = raw_score * speed_factor
        
        return np.clip(total, 0, 100)
    
    def get_driving_mode(self) -> str:
        """Restituisce il modo di guida attuale"""
        score = self.pushing_score()
        
        if score < 30:
            return "GESTIONE"
        elif score < 50:
            return "RITMO MEDIO"
        elif score < 70:
            return "SPINTA"
        else:
            return "MASSIMO ATTACCO"
    
    #########################################################
    #                       Grafici                         #
    #########################################################
    
    def plot_g_forces_map(self, figsize: Tuple[int, int] = (12, 10)):
        """
        Grafico 1: Mappa delle forze G (cerchio di aderenza)
        Mostra come il pilota usa il grip disponibile
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('ANALISI FORZE G E UTILIZZO GRIP', fontsize=16, fontweight='bold')
        
        # 1. Cerchio di aderenza (G plot)
        ax = axes[0, 0]
        scatter = ax.scatter(self.lateral_g(), self.longitudinal_g(), 
                            c=self.speed, cmap='plasma', s=10, alpha=0.6)
        
        # Cerchio teorico massimo
        theta = np.linspace(0, 2*np.pi, 100)
        max_g = 5.0
        ax.plot(max_g * np.cos(theta), max_g * np.sin(theta), 
                'r--', linewidth=2, alpha=0.3, label=f'Limite teorico ({max_g}G)')
        
        ax.set_xlabel('G Laterali', fontweight='bold')
        ax.set_ylabel('G Longitudinali', fontweight='bold')
        ax.set_title('Cerchio di Aderenza')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)
        ax.legend()
        plt.colorbar(scatter, ax=ax, label='Velocità (km/h)')
        ax.set_aspect('equal')
        
        # 2. G totali nel tempo
        ax = axes[0, 1]
        time = np.arange(len(self.total_g_xy()))
        ax.plot(time, self.total_g_xy(), color='#e74c3c', linewidth=1.5, label='G Totali')
        ax.fill_between(time, 0, self.total_g_xy(), alpha=0.3, color='#e74c3c')
        ax.set_xlabel('Campioni', fontweight='bold')
        ax.set_ylabel('G Totali', fontweight='bold')
        ax.set_title('G Totali nel Tempo')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 3. Utilizzo grip %
        ax = axes[1, 0]
        grip_usage = self.grip_usage_percent()
        ax.plot(time, grip_usage, color='#3498db', linewidth=1.5)
        ax.fill_between(time, 0, grip_usage, alpha=0.3, color='#3498db')
        ax.axhline(80, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Soglia rischio')
        ax.axhline(90, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Limite')
        ax.set_xlabel('Campioni', fontweight='bold')
        ax.set_ylabel('Utilizzo Grip (%)', fontweight='bold')
        ax.set_title('Percentuale Utilizzo Grip')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 4. Distribuzione G laterali vs longitudinali
        ax = axes[1, 1]
        ax.hist2d(self.lateral_g(), self.longitudinal_g(), 
                  bins=50, cmap='YlOrRd', alpha=0.8)
        ax.set_xlabel('G Laterali', fontweight='bold')
        ax.set_ylabel('G Longitudinali', fontweight='bold')
        ax.set_title('Distribuzione Forze G (Heatmap)')
        ax.grid(True, alpha=0.3)
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                fig.tight_layout()
        except Exception:
            pass
        return fig
    
    def plot_driver_inputs(self, figsize: Tuple[int, int] = (14, 8)):
        """
        Grafico 2: Input del pilota (throttle, brake, steering proxy)
        """
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        fig.suptitle('INPUT DEL PILOTA', fontsize=16, fontweight='bold')
        
        time = np.arange(len(self.throttle))
        
        # 1. Throttle
        ax = axes[0]
        ax.fill_between(time, 0, self.throttle, color='#2ecc71', alpha=0.7, label='Throttle')
        ax.set_ylabel('Throttle (%)', fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_title(f'Gas Spalancato: {self.full_throttle_percent():.1f}% del tempo')
        
        # 2. Brake
        ax = axes[1]
        ax.fill_between(time, 0, self.brake * 100, color='#e74c3c', alpha=0.7, label='Brake')
        ax.set_ylabel('Brake (On/Off)', fontweight='bold')
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_title(f'Tempo in Frenata: {self.brake_time_percent():.1f}%')
        
        # 3. G Laterali (proxy per steering)
        ax = axes[2]
        ax.plot(time, self.lateral_g(), color='#9b59b6', linewidth=1, label='G Laterali')
        ax.fill_between(time, 0, self.lateral_g(), where=self.lateral_g()>=0, 
                        alpha=0.3, color='#9b59b6', interpolate=True)
        ax.fill_between(time, 0, self.lateral_g(), where=self.lateral_g()<=0, 
                        alpha=0.3, color='#e67e22', interpolate=True)
        ax.set_xlabel('Campioni', fontweight='bold')
        ax.set_ylabel('G Laterali', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.axhline(0, color='k', linewidth=0.5)
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                fig.tight_layout()
        except Exception:
            pass
        return fig
    
    def plot_tire_management(self, figsize: Tuple[int, int] = (14, 6)):
        """
        Grafico 3: Gestione gomme
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle('GESTIONE GOMME', fontsize=16, fontweight='bold')
        
        # Determina la lunghezza per l'asse temporale
        telemetry_len = len(self.speed)
        time = np.arange(telemetry_len)
        
        # Gestione life (può essere scalare o array)
        life_data = np.asarray(self.life)
        if life_data.ndim == 0:
            life_plot = np.full(telemetry_len, self.life)
        else:
            life_plot = life_data
            if len(life_plot) != telemetry_len:
                # Se le lunghezze non coincidono, cerchiamo di interpolare o adattare
                life_plot = np.interp(np.linspace(0, len(life_plot)-1, telemetry_len), 
                                     np.arange(len(life_plot)), life_plot)

        # 1. Vita gomme nel tempo
        ax = axes[0]
        ax.plot(time, life_plot, color='#e74c3c', linewidth=2, marker='o', 
                markersize=2, label='Vita Gomme')
        ax.fill_between(time, 0, life_plot, alpha=0.3, color='#e74c3c')
        ax.axhline(20, color='orange', linestyle='--', alpha=0.5, label='Soglia critica')
        ax.set_xlabel('Campioni', fontweight='bold')
        ax.set_ylabel('Vita Gomme (%)', fontweight='bold')
        ax.set_title(f'Degrado: {self.tire_wear_total():.1f}%')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 100)
        
        # 2. Aggressivity Index
        ax = axes[1]
        agg_idx = self.aggressivity_index()
        ax.plot(time, agg_idx, color='#f39c12', linewidth=1.5)
        ax.fill_between(time, 0, agg_idx, alpha=0.3, color='#f39c12')
        ax.set_xlabel('Campioni', fontweight='bold')
        ax.set_ylabel('Aggressivity Index', fontweight='bold')
        ax.set_title('Stress su Gomme vs Degrado')
        ax.grid(True, alpha=0.3)
        
        # 3. Correlazione: Grip usage vs Tire life
        ax = axes[2]
        grip_usage = self.grip_usage_percent()
        
        # Assicuriamoci che grip_usage e life_plot abbiano la stessa dimensione
        scatter = ax.scatter(life_plot, grip_usage, c=self.speed, 
                            cmap='viridis', s=10, alpha=0.6)
        ax.set_xlabel('Vita Gomme (%)', fontweight='bold')
        ax.set_ylabel('Utilizzo Grip (%)', fontweight='bold')
        ax.set_title('Grip Usage vs Tire Life')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Velocità')
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                fig.tight_layout()
        except Exception:
            pass
        return fig
    
    def plot_pushing_analysis(self, figsize: Tuple[int, int] = (14, 10)):
        """
        Grafico 4: ANALISI PRINCIPALE - Sta spingendo o gestendo?
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        pushing_score = self.pushing_score()
        driving_mode = self.get_driving_mode()
        
        fig.suptitle(f'PUSHING ANALYSIS - Score: {pushing_score:.1f}/100 - {driving_mode}', 
                     fontsize=18, fontweight='bold')
        
        # 1. Pushing Score Gauge (grande, in alto)
        ax_gauge = fig.add_subplot(gs[0, :])
        self._plot_gauge(ax_gauge, pushing_score)
        
        # 2. Metriche chiave
        ax = fig.add_subplot(gs[1, 0])
        metrics = {
            'Grip Usage': np.mean(self.grip_usage_percent()),
            'Throttle': self.full_throttle_percent(),
            'Smoothness': (1 - self.smoothness_index()) * 100,
            'Corner Agg': np.clip(self.corner_aggression() * 0.5, 0, 100),
            'Brake Agg': np.clip(self.braking_aggression(), 0, 100)
        }
        
        colors = ['#3498db', '#2ecc71', '#9b59b6', '#e67e22', '#e74c3c']
        bars = ax.barh(list(metrics.keys()), list(metrics.values()), color=colors, alpha=0.7)
        ax.set_xlabel('Score', fontweight='bold')
        ax.set_title('Componenti Pushing Score')
        ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Aggiungi valori sulle barre
        for i, (bar, val) in enumerate(zip(bars, metrics.values())):
            ax.text(val + 2, i, f'{val:.1f}', va='center', fontweight='bold')
        
        # 3. Speed vs G totali
        ax = fig.add_subplot(gs[1, 1])
        scatter = ax.scatter(self.speed, self.total_g_xy(), 
                            c=self.throttle, cmap='RdYlGn', s=15, alpha=0.6)
        ax.set_xlabel('Velocità (km/h)', fontweight='bold')
        ax.set_ylabel('G Totali', fontweight='bold')
        ax.set_title('Velocità vs G Forces')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Throttle %')
        
        # 4. Brake vs Throttle scatter
        ax = fig.add_subplot(gs[1, 2])
        brake_points = self.brake == 1
        coast_points = (self.throttle < 10) & (self.brake == 0)
        throttle_points = self.throttle > 10
        
        if np.any(brake_points):
            ax.scatter(self.speed[brake_points], self.total_g_xy()[brake_points], 
                      color='red', s=10, alpha=0.5, label='Brake')
        if np.any(throttle_points):
            ax.scatter(self.speed[throttle_points], self.total_g_xy()[throttle_points], 
                      color='green', s=10, alpha=0.5, label='Throttle')
        if np.any(coast_points):
            ax.scatter(self.speed[coast_points], self.total_g_xy()[coast_points], 
                      color='gray', s=10, alpha=0.3, label='Coast')
        
        ax.set_xlabel('Velocità (km/h)', fontweight='bold')
        ax.set_ylabel('G Totali', fontweight='bold')
        ax.set_title('Fasi di Guida')
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Timeline pushing intensity
        ax = fig.add_subplot(gs[2, :])
        time = np.arange(len(self.speed))
        
        # Calcola pushing intensity istantaneo
        instant_push = self.grip_usage_percent()
        
        ax.fill_between(time, 0, instant_push, alpha=0.6, color='#e74c3c', label='Grip Usage %')
        ax.plot(time, self.speed / self.speed.max() * 100, 
                color='#3498db', linewidth=1.5, alpha=0.8, label='Velocità (norm)')
        
        # Evidenzia zone di massima spinta
        high_push = instant_push > 70
        if np.any(high_push):
            ax.fill_between(time, 0, 100, where=high_push, 
                           alpha=0.2, color='red', label='Zona Spinta Max')
        
        ax.set_xlabel('Campioni (Tempo)', fontweight='bold')
        ax.set_ylabel('Intensità (%)', fontweight='bold')
        ax.set_title('Timeline Intensità Spinta')
        ax.set_ylim(0, 100)
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        except Exception:
            pass
        return fig
    
    def _plot_gauge(self, ax, value):
        """Disegna un gauge per il pushing score"""
        # Sfondo gauge
        theta = np.linspace(0, np.pi, 100)
        
        # Arco di sfondo
        ax.plot(np.cos(theta), np.sin(theta), 'lightgray', linewidth=20, solid_capstyle='round')
        
        # Arco colorato in base al valore
        if value < 30:
            color = '#2ecc71'  # Verde
        elif value < 50:
            color = '#f39c12'  # Giallo
        elif value < 70:
            color = '#e67e22'  # Arancione
        else:
            color = '#e74c3c'  # Rosso
        
        # Disegna l'arco fino al valore
        theta_val = np.linspace(0, np.pi * (value/100), 50)
        ax.plot(np.cos(theta_val), np.sin(theta_val), color, linewidth=20, solid_capstyle='round')
        
        # Testo centrale
        ax.text(0, -0.3, f'{value:.1f}', ha='center', va='center', 
                fontsize=48, fontweight='bold', color=color)
        ax.text(0, -0.5, 'PUSHING SCORE', ha='center', va='center', 
                fontsize=14, color='gray')
        
        # Etichette
        ax.text(-1, 0, '0', ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(1, 0, '100', ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(-0.7, 0.7, 'GESTIONE', ha='center', va='center', fontsize=10, color='#2ecc71')
        ax.text(0.7, 0.7, 'SPINTA', ha='center', va='center', fontsize=10, color='#e74c3c')
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.7, 1.2)
        ax.axis('off')
        ax.set_aspect('equal')
    
    def plot_all(self, save_path: Optional[str] = None):
        """
        Genera tutti i grafici
        save_path: se specificato, salva i grafici invece di mostrarli
        """
        if not save_path:
            plt.close('all')
            
        print("Generazione grafici in corso...\n")
        
        # Grafico 1: G Forces
        print("1/4 - Analisi Forze G...")
        fig1 = self.plot_g_forces_map()
        if save_path:
            fig1.savefig(f'{save_path}_g_forces.png', dpi=150, bbox_inches='tight')
            print(f"   ✓ Salvato: {save_path}_g_forces.png")
        
        # Grafico 2: Driver Inputs
        print("2/4 - Input del Pilota...")
        fig2 = self.plot_driver_inputs()
        if save_path:
            fig2.savefig(f'{save_path}_inputs.png', dpi=150, bbox_inches='tight')
            print(f"   ✓ Salvato: {save_path}_inputs.png")
        
        # Grafico 3: Tire Management
        print("3/4 - Gestione Gomme...")
        fig3 = self.plot_tire_management()
        if save_path:
            fig3.savefig(f'{save_path}_tires.png', dpi=150, bbox_inches='tight')
            print(f"   ✓ Salvato: {save_path}_tires.png")
        
        # Grafico 4: Pushing Analysis
        print("4/4 - Analisi Spinta...")
        fig4 = self.plot_pushing_analysis()
        if save_path:
            fig4.savefig(f'{save_path}_pushing.png', dpi=150, bbox_inches='tight')
            print(f"   ✓ Salvato: {save_path}_pushing.png")
        
        print("\nTutti i grafici generati con successo!")
        
        if not save_path:
            plt.show()
            input("\n[INVIO per continuare...]")
            plt.close('all')
        else:
            plt.close('all')
    
    def print_summary(self):
        """Stampa un report testuale completo"""
        pushing_score = self.pushing_score()
        mode = self.get_driving_mode()
        
        print("=" * 60)
        print("TELEMETRY ANALYSIS REPORT".center(60))
        print("=" * 60)
        print()
        print(f"PUSHING SCORE: {pushing_score:.1f}/100")
        print(f"DRIVING MODE:  {mode}")
        print()
        print("-" * 60)
        print("METRICHE SEMPLICI:")
        print("-" * 60)
        print(f"  - Velocità Media:           {self.avg_speed():.1f} km/h")
        print(f"  - Velocità Massima:         {self.max_speed():.1f} km/h")
        print(f"  - Max G Laterali:           {np.max(np.abs(self.lateral_g())):.2f} G")
        print(f"  - Max G Longitudinali:      {np.max(self.longitudinal_g()):.2f} G")
        print(f"  - Max G Totali:             {np.max(self.total_g_xy()):.2f} G")
        print(f"  - Tempo in Frenata:         {self.brake_time_percent():.1f}%")
        print(f"  - Gas Spalancato:           {self.full_throttle_percent():.1f}%")
        print(f"  - Degrado Gomme:            {self.tire_wear_total():.1f}%")
        print()
        print("-" * 60)
        print("METRICHE COMPLESSE:")
        print("-" * 60)
        print(f"  • Utilizzo Grip Medio:      {np.mean(self.grip_usage_percent()):.1f}%")
        print(f"  • Smoothness Index:         {self.smoothness_index():.3f}")
        print(f"  • Aggressività Curve:       {self.corner_aggression():.1f}")
        print(f"  • Aggressività Frenata:     {self.braking_aggression():.1f}")
        print(f"  • Velocità in Curva:        {self.corner_speed_index():.1f} km/h")
        print(f"  • Tire Stress Score:        {self.tire_stress_score():.1f}")
        print()
        print("=" * 60)
        print()
        
        # Interpretazione
        if pushing_score < 30:
            print("INTERPRETAZIONE:")
            print("   Il pilota sta GESTENDO. Guida conservativa,")
            print("   probabilmente per preservare gomme o macchina.")
        elif pushing_score < 50:
            print("INTERPRETAZIONE:")
            print("   Ritmo di GARA. Il pilota spinge ma in modo controllato.")
        elif pushing_score < 70:
            print("INTERPRETAZIONE:")
            print("   Il pilota sta SPINGENDO. Vicino al limite della macchina.")
        else:
            print("INTERPRETAZIONE:")
            print("   MASSIMO ATTACCO! Probabile giro di qualifica o sorpasso.")
        print()
