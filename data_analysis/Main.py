from  Model.Curve import Curve
from utils.CurveDetector import CurveDetector
import os
import numpy as np



def media(curves):
    mammt = []
    for curve in curves:
        mammt.append(curve.pushing_score())
    
  
    print("mammt:")
    print(f"\t\tMedia : {np.asarray(mammt).mean()}")
    print(f"\t\tStd : {np.asarray(mammt).std()}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

def main():
    """Funzione main del progetto Iot Zecconi e Gagliotti Qualifying"""
 

    curve_detector = CurveDetector("data/2025-main/Abu Dhabi Grand Prix/Qualifying/NOR/1_tel.json", "data/2025-main/Abu Dhabi Grand Prix/Race/corners.json")
    curves = curve_detector.calcolo_curve()
    for i in range(0 , len(curves)):
    # #     #curves[i].plot_telemetry()
    # #     curves[i].plot_g_forces_map()
    # #     curves[i].plot_driver_inputs()
    # #     curves[i].plot_tire_management()
    # #     curves[i].plot_pushing_analysis()
        curves[i].plot_all()
        
    curve_detector.grafico(curves,False)
    curve_detector.plot_curve_trajectories(curves)
    media(curves)

if __name__ == "__main__":
    main()
                    
                        