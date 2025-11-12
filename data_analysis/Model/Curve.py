import math
from typing import List

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


    def print(self):
        print(f"[!] apex:{self.apex_dist}; start: {self.distance[0]} -> end: {self.distance[-1]}")

    # --- Tempo & distanza ---
    def time_in_curve(self) -> float:
        return self.time[-1] - self.time[0] 

    def distance_in_curve(self) -> float:
        return self.distance[-1] - self.distance[0] 

    def speed_average(self) -> float:
        return sum(self.speed) / len(self.speed) 

    def entry_speed_average(self, fraction: float = 0.05) -> float:
        amount = max(1, int(len(self.speed) * fraction))
        entry_speeds = self.speed[:amount]
        return sum(entry_speeds) / len(entry_speeds) 

    def exit_speed_average(self, fraction: float = 0.05) -> float:
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


    #  --- Traiettoria ---
    # def curvature(self) -> float:
    #     if len(self.x) < 2 or len(self.y) < 2:
    #         return 0.0
    #     # semplice variazione di direzione tra punti consecutivi
    #     total_delta = sum(abs((self.y[i+1]-self.y[i]) / (self.x[i+1]-self.x[i]+1e-6)) for i in range(len(self.x)-1))
    #     return total_delta / (len(self.x)-1)

    #  --- Efficienza dinamica ---
    # def energy_input(self) -> float:
    #     return sum(t*s for t, s in zip(self.throttle, self.speed)) if self.throttle and self.speed else 0.0

    # def energy_lost_brake(self) -> float:
    #     return sum(b * ax for b, ax in zip(self.brake, self.acc_x)) if self.brake and self.acc_x else 0.0

    # # --- Indicatori sintetici ---
    # def efficiency(self) -> float:
    #     dt = self.time_in_curve()
    #     return (self.exit_speed_average()**2 - self.entry_speed_average()**2) / dt 

