from  Model.Curve import Curve
from utils.CurveDetector import CurveDetector
import os

def main():
    """Funzione main del progetto Iot Zecconi e Gagliotti"""
 

    curve_detector = CurveDetector("data/2025-main/Japanese Grand Prix/Race/ALB/1_tel.json", "data/2025-main/Japanese Grand Prix/Race/corners.json")
    curves = curve_detector.calcolo_curve()
    #for i in range(0 , len(curves)):
        #print(len(curves[i].time))
        # curves[i].plot_trajectory_speed()
        # curves[i].plot_curvature()
        # curves[i].plot_vehicle_stability()
        # curves[i].plot_efficiency_profile()
        # curves[i].plot_energy_input_profile(use_time=True)
        # curves[i].plot_energy_lost_brake_profile(use_time=True)
        # curves[i].plot_aggressiveness_profile(use_time=True)
        # curves[i].plot_fluidity_profile()
        # curves[i].plot_trail_braking_profile(use_time=True)
        # curves[i].plot_grip_usage_profile(use_time=True)
        #print(Curve.classify_driver_style(curves[i]))
        
    curve_detector.grafico(curves,False)
    curve_detector.plot_curve_trajectories(curves)

if __name__ == "__main__":
    main()
