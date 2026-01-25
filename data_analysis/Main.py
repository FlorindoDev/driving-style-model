from  Model.Curve import Curve
from utils.CurveDetector import CurveDetector


def main():
    """Funzione main del progetto Iot Zecconi e Gagliotti"""
    print("Ciao dal main!")
<<<<<<< HEAD
    curve_detector = CurveDetector("data/48_telL_SLOW.json", "data/corners_L.json")
=======
    curve_detector = CurveDetector("data/21_telS_OUTLAP.json", "data/corners_S.json")
>>>>>>> bb2f3673385c3c01a5e8f7fa77fcc8eb3cab3334
    curves = curve_detector.calcolo_curve()
    #for i in range(0 , len(curves)):
       
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
<<<<<<< HEAD
        #print(Curve.classify_driver_style(curves[i]))
=======
        print(Curve.classify_driver_style(curves[i]))
>>>>>>> bb2f3673385c3c01a5e8f7fa77fcc8eb3cab3334
        
    curve_detector.grafico(curves,False)
    curve_detector.plot_curve_trajectories(curves)

if __name__ == "__main__":
    main()
