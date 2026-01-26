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

