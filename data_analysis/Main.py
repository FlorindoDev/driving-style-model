from  Model.Curve import Curve
from utils.CurveDetector import CurveDetector


def main():
    """Funzione main del progetto Iot Zecconi e Gagliotti"""
    print("Ciao dal main!")
    curve_detector = CurveDetector("data/24_telS_NOR.json", "data/corners_S.json")
    curves = curve_detector.calcolo_curve()
    for i in range(0 , len(curves)):
        curves[i].print();
        curves[i].plot_trajectory_speed()      # XY colorato per speed
        curves[i].plot_curvature()             # κ(s)
        curves[i].plot_efficiency_profile()
        curves[i].plot_stability_profile()
        curves[i].plot_energy_input_profile(use_time=True)
        curves[i].plot_energy_lost_brake_profile(use_time=True)
        curves[i].plot_aggressiveness_profile(use_time=True)
        curves[i].plot_fluidity_profile(use_time=True)
        print(Curve.classify_driver_style(curves[i]))
        
    curve_detector.grafico(curves,False)
    curve_detector.plot_curve_trajectories(curves)

if __name__ == "__main__":
    main()
