import json
import matplotlib.pyplot as plt
from Model.Curve import Curve
from .utils_for_array import *

class CurveDetector:
    # ================== PARAMETRI DA TARARE (default) ==================
    ACC_ENTER_THR_DEFAULT = 3.0    # entra in curva se |acc_y| > di questo
    ACC_EXIT_THR_DEFAULT  = 2.5    # esci se |acc_y| < di questo (leggermente più basso = isteresi)
    SMOOTH_WIN_DEFAULT    = 3      # smoothing su acc_y
    MIN_SAMPLES_IN_CURVE_DEFAULT = 5   # minimo punti per dire che è davvero curva
    # ===================================================================

    def __init__(
        self,
        telemetry_filename: str,
        corners_filename: str,
        acc_enter_thr: float = None,
        acc_exit_thr: float = None,
        smooth_win: int = None,
        min_samples_in_curve: int = None,
    ):
       
        self.ACC_ENTER_THR = acc_enter_thr if acc_enter_thr is not None else self.ACC_ENTER_THR_DEFAULT
        self.ACC_EXIT_THR  = acc_exit_thr  if acc_exit_thr  is not None else self.ACC_EXIT_THR_DEFAULT
        self.SMOOTH_WIN    = smooth_win    if smooth_win    is not None else self.SMOOTH_WIN_DEFAULT
        self.MIN_SAMPLES_IN_CURVE = (
            min_samples_in_curve if min_samples_in_curve is not None else self.MIN_SAMPLES_IN_CURVE_DEFAULT
        )

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

    # -------------------------------------------------------------

    def isApexInInterval(self, dist_array, apex_distances, index_end, index_start, curve_number):
        if dist_array[index_start] > apex_distances[curve_number]:
            return False

        if dist_array[index_end] < apex_distances[curve_number]:
            return False

        return True

    def getBounds(self, curve_number, apex_point, prev_apex, next_apex, n_corners):
        DEFAULT_FIRST_DISTANCE_CURVE = 150
        DEFAULT_LAST_DISTANCE_CURVE = 100

        if curve_number == 0:
            lower_bound = apex_point - DEFAULT_FIRST_DISTANCE_CURVE
        else:
            lower_bound = 0.5 * (prev_apex + apex_point)

        if curve_number == n_corners - 1:
            upper_bound = apex_point + DEFAULT_LAST_DISTANCE_CURVE
        else:
            upper_bound = 0.5 * (apex_point + next_apex)

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
                    if not self.isApexInInterval(self.tel_dist, self.corner_distances, idx, curve_start_idx, curve_number):
                        in_curve = False
                        curve_start_idx = None
                    else:
                        curve_end_idx = idx
                        break

            # se sono arrivato alla fine finestra e sono ancora "in curva"
            if in_curve and curve_end_idx is None:
                curve_end_idx = end_win_idx

        return curve_start_idx, curve_end_idx

    def calcolo_curve(self) -> list:
        detected_corners = []

        n_corners = len(self.corner_distances)

        for i in range(n_corners):

            current_apex_point = self.corner_distances[i]
            prev_apex = 0
            next_apex = 0

            if (i > 0):
                prev_apex = self.corner_distances[i - 1]
            if (i < n_corners - 1):
                next_apex = self.corner_distances[i + 1]

            lower_bound, upper_bound = self.getBounds(i, current_apex_point, prev_apex, next_apex, n_corners)

            start_win_idx = find_frist_value(self.tel_dist, lower_bound)
            end_win_idx = find_last_value(self.tel_dist, upper_bound)

            curve_start_idx, curve_end_idx = self.getCurveWindow(start_win_idx, end_win_idx, i)

            # salvo solo se ho trovato qualcosa di sensato
            if curve_start_idx is not None and curve_end_idx is not None:
                if (curve_end_idx - curve_start_idx + 1) >= self.MIN_SAMPLES_IN_CURVE:

                    curve = Curve(
                        corner_id=self.corner_numbers[i],
                        apex_dist=current_apex_point,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
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

    def grafico(self, detected_corners):
        # --- GRAFICO ---
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

        for c in detected_corners:
            # verticale del punto TEORICO (apice curva)
            theo_idx = find_frist_value(self.tel_dist, c.apex_dist)
            ax1.axvline(self.time[theo_idx], color='black', linestyle='--', linewidth=1, alpha=0.5)

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
        plt.show()


# Esempio d'uso:
# detector = CurveDetector("18_tel.json", "corners_S.json")
# curves = detector.calcolo_curve()
# detector.grafico(curves)
