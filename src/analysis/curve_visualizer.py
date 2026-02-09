from src.analysis.CurveDetector import CurveDetector
from src.models.dataset_loader import download_raw_telemetry_from_hf
from dataclasses import dataclass


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class VisualizerConfig:
    """Configuration for curve visualization."""
    
    # ----- Paths -----
    telemetry_path: str = "data/2025-main/Australian Grand Prix/Qualifying/NOR/19_tel.json"
    corners_path: str = "data/2025-main/Australian Grand Prix/Race/corners.json"
    
    # ----- Hugging Face -----
    download_from_hf: bool = True  # True = download raw telemetry from HF
    raw_data_subfolder: str = "2025-main"  # Subfolder to download (2024-main or 2025-main)
    
    # ----- Visualization -----
    show_track: bool = False    # Show track overview with curves
    show_score: bool = False     # Show pushing score


CONFIG = VisualizerConfig()


# =============================================================================
# MAIN
# =============================================================================
def main(config: VisualizerConfig = CONFIG):
    """Funzione main per visualizzazione curve."""
    
    print("=" * 60)
    print("Curve Visualizer")
    print("=" * 60)
    
    # Download raw data from HF if configured
    if config.download_from_hf:
        print("\n[1/3] Downloading raw telemetry from Hugging Face...")
        download_raw_telemetry_from_hf(subfolder=config.raw_data_subfolder)
    else:
        print("\n[1/3] Using local telemetry data...")
    
    # Detect curves
    print("\n[2/3] Detecting curves...")
    curve_detector = CurveDetector(config.telemetry_path, config.corners_path)
    curves = curve_detector.calcolo_curve()
    print(f"Detected {len(curves)} curves")
    
    # Visualize
    print("\n[3/3] Visualizing...")
    curve_detector.grafico(curves, config.show_track)
    curve_detector.plot_curve_trajectories(curves)

    if config.show_score:
        for curve in curves:
            curve.plot_pushing_analysis()
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
