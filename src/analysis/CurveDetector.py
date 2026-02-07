import json
import os
import matplotlib.pyplot as plt
from src.analysis.Curve import Curve
from src.analysis.utils_for_array import *
import math
import os




class CurveDetector:
    # ================== PARAMETRI DA TARARE (default) ==================
    ACC_ENTER_THR_DEFAULT = 3.0    # entra in curva se |acc_y| > di questo
    ACC_EXIT_THR_DEFAULT  = 2.5    # esci se |acc_y| < di questo (leggermente più basso = isteresi)
    SMOOTH_WIN_DEFAULT    = 3      # smoothing su acc_y
    MIN_SAMPLES_IN_CURVE_DEFAULT = 5   # minimo punti per dire che è davvero curva
    MAX_SAMPLES_IN_CURVE_DEFAULT = 25 # grandezza massima curva
    # ===================================================================

    def __init__(
        self,
        telemetry_filename: str,
        corners_filename: str,
        acc_enter_thr: float = None,
        acc_exit_thr: float = None,
        smooth_win: int = None,
        min_samples_in_curve: int = None,
        max_samples_in_curve: int = None,
    ):
       
        self.ACC_ENTER_THR = acc_enter_thr if acc_enter_thr is not None else self.ACC_ENTER_THR_DEFAULT
        self.ACC_EXIT_THR  = acc_exit_thr  if acc_exit_thr  is not None else self.ACC_EXIT_THR_DEFAULT
        self.SMOOTH_WIN    = smooth_win    if smooth_win    is not None else self.SMOOTH_WIN_DEFAULT
        self.MIN_SAMPLES_IN_CURVE = (
            min_samples_in_curve if min_samples_in_curve is not None else self.MIN_SAMPLES_IN_CURVE_DEFAULT
        )
        self.MAX_SAMPLES_IN_CURVE = (
            max_samples_in_curve if max_samples_in_curve is not None else self.MAX_SAMPLES_IN_CURVE_DEFAULT
        )

        
        self.telemetry_filename = telemetry_filename
        self.corners_filename = corners_filename

        # --- Tire Info Extraction ---
        self.compound = "UNKNOWN"
        self.tire_life = 0
        self.stint = 0
        
        # --- 1) carico telemetria ---
        with open(telemetry_filename, "r") as f:
            tel_data = json.load(f)

        tel = tel_data["tel"]
        self.rpm = tel["rpm"]
        self.speed = tel["speed"]
        self.gear = tel["gear"]
        self.acc_x = tel["acc_x"]
        self.acc_z = tel["acc_z"]
        self.x = tel["x"]
        self.y = tel["y"]
        self.z = tel["z"]
        self.time = tel["time"]
        self.acc_y = tel["acc_y"]
        self.tel_dist = tel["distance"]
        self.throttle = tel["throttle"]
        self.brake = tel["brake"]
        

        # --- 2) carico corner map ---
        with open(corners_filename, "r") as f:
            corner_map = json.load(f)

        self.corner_numbers = corner_map["CornerNumber"]
        self.corner_distances = corner_map["Distance"]
        self.corner_X = corner_map["X"]
        self.corner_Y = corner_map["Y"]


    # -------------------------------------------------------------


    def distanza(self, p1, p2):
        if len(p1) != len(p2):
            raise ValueError("I due punti devono avere la stessa dimensione")
        somma = 0

        for a, b in zip(p1, p2):
            somma += (b - a) ** 2
        return math.sqrt(somma)

    def nearest_point_index(self,xs, ys, cx, cy):
        """Ritorna (idx, dist) del punto (xs[idx], ys[idx]) più vicino a (cx, cy)."""
        best_idx = None
        best_dist = float("inf")
        for idx, (x, y) in enumerate(zip(xs, ys)):
            d = self.distanza((x, y), (cx, cy))
            if d < best_dist:
                best_dist = d
                best_idx = idx
        return best_idx, best_dist

    def compaund_and_life(self):

        try:
            dir_path = os.path.dirname(self.telemetry_filename)
            base_name = os.path.basename(self.telemetry_filename)
            # Extract lap number from filename (e.g., "12_tel.json" -> 12)
            parts = base_name.split("_")
            if parts[0].isdigit():
                current_lap = int(parts[0]) - 1
               
                laptimes_path = os.path.join(dir_path, "laptimes.json")
                if os.path.exists(laptimes_path):
                    with open(laptimes_path, "r") as f:
                        laptimes_data = json.load(f)
                        
                    if "lap" in laptimes_data:
                        
                        if current_lap != -1:
                            if "compound" in laptimes_data:
                                self.compound = laptimes_data["compound"][current_lap]
                            if "life" in laptimes_data:
                                self.tire_life = laptimes_data["life"][current_lap]
                            if "stint" in laptimes_data:
                                self.stint = laptimes_data["stint"][current_lap]

        except Exception as e:
            pass



    def isApexInInterval(self, index_end, index_start):
        if index_start > self.index_center_current_corner:
            return False

        if index_end < self.index_center_current_corner:
            return False

        return True

    def getBounds(self, curve_number, corner_point, prev_corner, next_corner, n_corners):
        DEFAULT_FIRST_DISTANCE_CURVE = 150
        DEFAULT_LAST_DISTANCE_CURVE = 250

        if curve_number == 0:
            lower_bound = corner_point - DEFAULT_FIRST_DISTANCE_CURVE
        else:
            lower_bound = 0.5 * (prev_corner + corner_point)

        if curve_number == n_corners - 1:
            upper_bound = corner_point + DEFAULT_LAST_DISTANCE_CURVE
        else:
            upper_bound = 0.5 * (corner_point + next_corner)

        return lower_bound, upper_bound

    def getCurveWindow(self, start_win_idx, end_win_idx, curve_number):
        in_curve = False
        curve_start_idx = None
        curve_end_idx = None

        for idx in range(start_win_idx, end_win_idx + 1):
            acc_smooth = avg_in_window(self.acc_y, idx, self.SMOOTH_WIN)
            if not in_curve:
                if acc_smooth >= self.ACC_ENTER_THR:
                    in_curve = True
                    curve_start_idx = idx
            else:
                if acc_smooth <= self.ACC_EXIT_THR:
                    if not self.isApexInInterval( idx, curve_start_idx):
                        in_curve = False
                        curve_start_idx = None
                    else:
                        curve_end_idx = idx
                        break
         

            # se sono arrivato alla fine finestra e sono ancora "in curva"
            if in_curve and curve_end_idx is None:
                curve_end_idx = end_win_idx

        if curve_start_idx is not None:
            if not self.isApexInInterval(curve_end_idx, curve_start_idx):
                curve_end_idx = None
                curve_start_idx = None 
        
        

        return curve_start_idx, curve_end_idx

    def calcolo_curve(self) -> list:

        self.compaund_and_life()

        detected_corners = []

        n_corners = len(self.corner_distances)

        for i in range(0,n_corners):

            self.index_center_current_corner, _ = self.nearest_point_index(self.x,self.y ,self.corner_X[i],self.corner_Y[i])
            current_corner_distance = self.tel_dist[self.index_center_current_corner]
            
            prev_center_curve = 0
            next_center_curve = 0
            
            if (i > 0):
                prev_index_center_current_corner, _ = self.nearest_point_index(self.x,self.y ,self.corner_X[i-1],self.corner_Y[i-1])
                prev_center_curve = self.tel_dist[prev_index_center_current_corner]
            if (i < n_corners - 1):
                next_index_center_current_corner, _ = self.nearest_point_index(self.x,self.y ,self.corner_X[i+1],self.corner_Y[i+1])
                next_center_curve = self.tel_dist[next_index_center_current_corner]

            lower_bound, upper_bound = self.getBounds(i, current_corner_distance, prev_center_curve, next_center_curve, n_corners)


            start_win_idx = find_frist_value(self.tel_dist, lower_bound)
            end_win_idx = find_last_value(self.tel_dist, upper_bound)

            curve_start_idx, curve_end_idx = self.getCurveWindow(start_win_idx, end_win_idx, i)


            if curve_start_idx is not None and curve_end_idx is not None:
                lenght_curve = curve_end_idx - curve_start_idx + 1
                if lenght_curve >= self.MIN_SAMPLES_IN_CURVE:
                  
                    distance_from_start = self.index_center_current_corner - curve_start_idx
                    distance_from_end   = curve_end_idx - self.index_center_current_corner

                    MARGIN_BEFORE = 25   
                    MARGIN_AFTER  = 25

                    # se la curva inizia troppo lontano dall'apice → avvicino lo start
                    if distance_from_start > MARGIN_BEFORE:
                        curve_start_idx = self.index_center_current_corner - MARGIN_BEFORE

                    # se la curva finisce troppo lontano dall'apice → avvicino l'end
                    if distance_from_end > MARGIN_AFTER:
                        curve_end_idx = self.index_center_current_corner + MARGIN_AFTER

                            

                    curve = Curve(
                        corner_id=self.corner_numbers[i],
                        current_corner_dist=current_corner_distance,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        compound=self.compound,
                        life= self.tire_life,
                        stint= self.stint,
                        time=self.time[curve_start_idx:curve_end_idx],
                        rpm=self.rpm[curve_start_idx:curve_end_idx],
                        speed=self.speed[curve_start_idx:curve_end_idx],
                        throttle=self.throttle[curve_start_idx:curve_end_idx],
                        brake=self.brake[curve_start_idx:curve_end_idx],
                        distance=self.tel_dist[curve_start_idx:curve_end_idx],
                        acc_x=self.acc_x[curve_start_idx:curve_end_idx],
                        acc_y=self.acc_y[curve_start_idx:curve_end_idx],
                        acc_z=self.acc_z[curve_start_idx:curve_end_idx],
                        x=self.x[curve_start_idx:curve_end_idx],
                        y=self.y[curve_start_idx:curve_end_idx],
                        z=self.z[curve_start_idx:curve_end_idx],
                    )

                    detected_corners.append(curve)

        return detected_corners

    def grafico(self, detected_corners, block=True):
     
        fig, ax1 = plt.subplots(figsize=(12, 5))

        # acc_y
        ax1.plot(self.time, self.acc_y, color='tab:blue', label='Acc Y (m/s²)')
        ax1.set_xlabel('Tempo (s)')
        ax1.set_ylabel('Acc Y (m/s²)', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # secondo asse: throttle/brake
        ax2 = ax1.twinx()
        ax2.plot(self.time, self.throttle, color='tab:red', alpha=0.5, label='Throttle (%)')
        ax2.plot(self.time, self.brake, color='tab:green', alpha=0.5, label='Brake (%)')
        ax2.set_ylabel('Throttle / Brake (%)', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        i=0
        for c in detected_corners:
            # verticale del punto TEORICO (apice curva)
           
            theo_idx, _ = self.nearest_point_index(self.x, self.y, self.corner_X[i], self.corner_Y[i])
            ax1.axvline(self.time[theo_idx], color='black', linestyle='--', linewidth=1, alpha=0.5)
            i+=1
            # finestra di ricerca [lower_bound, upper_bound]
            left_idx = find_frist_value(self.tel_dist, c.lower_bound)
            right_idx = find_last_value(self.tel_dist, c.upper_bound)
            ax1.axvspan(self.time[left_idx], self.time[right_idx], color='gray', alpha=0.05)

            # evidenzia la curva trovata (se presente)
            if len(c.time) > 0:
                ax1.axvspan(c.time[0], c.time[-1], color='orange', alpha=0.2)

        # legenda
        l1, lab1 = ax1.get_legend_handles_labels()
        l2, lab2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, lab1 + lab2, loc='upper right')

        ax1.set_title("Raffinamento curve con finestra per distanza")
        fig.tight_layout()
        plt.show(block=block)

    def plot_curve_trajectories(self, detected_corners, show_apex=True, block=True):
        """
        Disegna la traiettoria XY del giro completo e, sopra,
        evidenzia per ogni curva il tratto effettivo percorso dal pilota.

        Parametri
        ----------
        detected_corners : list[Curve]
            Lista di oggetti Curve restituiti da calcolo_curve().
        show_apex : bool
            Se True, marca anche il punto dell'apice per ogni curva.
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        # Disegno l'intero giro in grigio chiaro
        ax.plot(self.x, self.y, linewidth=1, alpha=0.3, label="Giro completo")

        # Per ogni curva, evidenzio la traiettoria dentro la curva
        
        for c in detected_corners:

            # Traiettoria effettiva nella curva (già slice dell'intera telemetria)
            ax.plot(c.x, c.y, linewidth=2, label=f"Curva {c.corner_id}")

           
            
        i=0
        for curve in self.corner_X:
            try:
                apex_idx_local = find_frist_value(c.distance, c.current_corner_dist)
                apex_x = c.x[apex_idx_local]
                apex_y = c.y[apex_idx_local]
                apex_x = self.corner_X[i]
                apex_y = self.corner_Y [i]
                i+=1
                ax.scatter(apex_x, apex_y, s=40, marker="x", color="red")
                ax.text(
                    apex_x,
                    apex_y,
                    f"{i}",
                    fontsize=8,
                    color="red",
                    ha="left",
                    va="bottom",
                )
            except Exception:
                # Se qualcosa va storto, semplicemente non disegno il marker
                pass
        # i=0
        # for curve in self.corner_distances:
        #     try:
        #         apex_idx_local = find_frist_value(self.tel_dist, curve)
        #          print(f"{self.tel_dist[apex_idx_local]} distanza pilota")
        #          print(f"{self.tel_dist[apex_idx_local]} distanza pilota")
        #         apex_x = self.x[apex_idx_local]
        #         apex_y = self.y[apex_idx_local]
        #          print(f"n:{apex_x}, {apex_y}")
        #         i+=1
        #         ax.scatter(apex_x, apex_y, s=40, marker="x", color="blue")
        #         ax.text(
        #             apex_x,
        #             apex_y,
        #             f"{i}",
        #             fontsize=8,
        #             color="red",
        #             ha="left",
        #             va="bottom",
        #         )
        #     except Exception:
        #         # Se qualcosa va storto, semplicemente non disegno il marker
        #         pass
        
        
        # --- nel tuo plotting ---
        i = 0
        for cx, cy in zip(self.corner_X, self.corner_Y):
            try:
                apex_idx_local, dmin = self.nearest_point_index(self.x, self.y, cx, cy)

                # print(f"{apex_idx_local} indice (dist={dmin})")
                apex_x = self.x[apex_idx_local]
                apex_y = self.y[apex_idx_local]
                # print(f"n:{apex_x}, {apex_y}")

                i += 1
                ax.scatter(apex_x, apex_y, s=40, marker="x", color="green")
                ax.text(apex_x, apex_y, f"{i}", fontsize=8, color="red", ha="left", va="bottom")
            except Exception:
                pass
        
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Traiettoria in curva (XY)")

        # Scala uguale sugli assi per non deformare la pista
        ax.set_aspect("equal", adjustable="box")


        plt.tight_layout()
        plt.show(block=block)



