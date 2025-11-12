from  Model.Curve import Curve
from utils.CurveDetector import CurveDetector


def main():
    """Funzione main del progetto Iot Zecconi e Gagliotti"""
    print("Ciao dal main!")
    curve_detector = CurveDetector("data/4_tel.json", "data/corners_M.json")
    curves = curve_detector.calcolo_curve()
    for i in range(0 , len(curves)):
        curves[i].print();
    
    curve_detector.grafico(curves)

if __name__ == "__main__":
    main()
