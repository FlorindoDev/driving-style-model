
import os
import sys
import json
import tempfile
import traceback

import numpy as np
from flask import Blueprint, request, jsonify

# ── Ensure project root is importable ────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

analysis_bp = Blueprint('analysis', __name__)

# ── Cluster labels (must match train.py / evaluate.py) ───────────
CLUSTER_NAMES = {
    0: "Fast Pace",
    1: "Pushing",
    2: "Slow Pace",
    3: "Saving",
}

CLUSTER_COLORS = {
    0: "#3498db",   # blue
    1: "#e74c3c",   # red
    2: "#f39c12",   # amber
    3: "#2ecc71",   # green
}

# ── Paths (relative to project root) ─────────────────────────────
DATASET_PATH   = os.path.join(PROJECT_ROOT, 'data', 'dataset',
                              'normalized_dataset_2024_2025.npz')
WEIGHTS_PATH   = os.path.join(PROJECT_ROOT, 'src', 'models', 'weights',
                              'VAE_32z_weights.pth')
CENTROIDS_PATH = os.path.join(PROJECT_ROOT, 'src', 'models', 'weights',
                              'kmeans_centroids.npy')

# ── Lazy-loaded analysis resources ────────────────────────────────
_analysis_cache: dict = {}


def _get_analysis_resources():
    """Load VAE model, dataset stats and KMeans centroids (once)."""
    if 'model' in _analysis_cache:
        return _analysis_cache

    import torch
    from sklearn.cluster import KMeans
    from src.analysis.dataset_normalization import load_normalized_data
    from src.models.VAE import VAE

    # If dataset doesn't exist locally, try downloading
    if not os.path.exists(DATASET_PATH):
        try:
            from src.models.dataset_loader import download_dataset_from_hf
            download_dataset_from_hf(
                filename="normalized_dataset_2024_2025.npz",
                filepath="data/dataset/"
            )
        except Exception as e:
            print(f"Failed to auto-download dataset: {e}")
            traceback.print_exc()
            # Don't silence it, let it fail so user knows why analysis won't work
            pass 

    print("📊  Loading analysis resources (first time)…")
    data, mask, mean, std, columns = load_normalized_data(DATASET_PATH)

    model = VAE(data.shape[1], latent_dim=32)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.eval()

    centroids = np.load(CENTROIDS_PATH)
    kmeans = KMeans(n_clusters=4, random_state=0, n_init=1)
    kmeans.cluster_centers_ = centroids
    kmeans._n_threads = 1

    _analysis_cache.update(
        model=model, kmeans=kmeans, data=data, mean=mean, std=std
    )
    print("✅  Analysis resources loaded")
    return _analysis_cache


# ── Helper: smooth with numpy ────────────────────────────────────
def _smooth(arr, win=7):
    kernel = np.ones(win) / win
    return np.convolve(arr, kernel, mode='same')


# ── Convert FastF1 telemetry → CurveDetector JSON ────────────────
def _compute_accelerations(x, y, z, speed_kmh, distance, time_s):
    """
    Compute car-frame accelerations from FastF1 position + speed data.
    Returns lists: acc_x (longitudinal m/s²), acc_y (lateral m/s²), acc_z.
    """
    x  = np.asarray(x, dtype=float)
    y  = np.asarray(y, dtype=float)
    z  = np.asarray(z, dtype=float)
    t  = np.asarray(time_s, dtype=float)
    spd = np.asarray(speed_kmh, dtype=float) / 3.6   # → m/s
    dist = np.asarray(distance, dtype=float)

    # ── Heading from position ──
    dx = np.gradient(x)
    dy = np.gradient(y)
    heading = np.unwrap(np.arctan2(dy, dx))
    heading = _smooth(heading, 9)

    # ── Curvature  κ = dθ / ds ──
    ds = np.gradient(dist)
    ds[np.abs(ds) < 1e-6] = 1e-6
    dheading = np.gradient(heading)
    curvature = dheading / ds

    # ── Lateral acceleration  a_lat = v² · κ ──
    acc_lat = spd ** 2 * curvature
    acc_lat = _smooth(acc_lat, 5)

    # ── Longitudinal acceleration  a_lon = dv / dt ──
    dt = np.gradient(t)
    dt[np.abs(dt) < 1e-6] = 1e-6
    acc_lon = np.gradient(spd) / dt
    acc_lon = _smooth(acc_lon, 5)

    # ── Vertical (simple) ──
    vz = np.gradient(z, t)
    acc_vert = np.gradient(vz, t)

    return acc_lon.tolist(), acc_lat.tolist(), acc_vert.tolist()


def _fastf1_to_tel_json(telemetry_df):
    """Build the dict that CurveDetector expects from a *_tel.json file."""
    time_col = telemetry_df['Time']
    if hasattr(time_col.iloc[0], 'total_seconds'):
        time_s = [t.total_seconds() for t in time_col]
    else:
        time_s = list(time_col)

    x = telemetry_df['X'].tolist()
    y = telemetry_df['Y'].tolist()
    z = telemetry_df['Z'].tolist() if 'Z' in telemetry_df.columns \
        else [0.0] * len(x)
    speed    = telemetry_df['Speed'].tolist()
    rpm      = telemetry_df['RPM'].tolist()
    gear     = telemetry_df['nGear'].tolist()
    throttle = telemetry_df['Throttle'].tolist()
    brake    = [float(b) for b in telemetry_df['Brake'].tolist()]
    distance = telemetry_df['Distance'].tolist()

    acc_x, acc_y, acc_z = _compute_accelerations(
        x, y, z, speed, distance, time_s
    )

    return {
        "tel": dict(
            rpm=rpm, speed=speed, gear=gear,
            acc_x=acc_x, acc_y=acc_y, acc_z=acc_z,
            x=x, y=y, z=z,
            time=time_s, distance=distance,
            throttle=throttle, brake=brake,
        )
    }


def _circuit_info_to_corners(circuit_info, telemetry_df):
    """Build corners.json dict from FastF1 CircuitInfo."""
    corners = circuit_info.corners
    nums = corners['Number'].tolist()
    dists = corners['Distance'].tolist()

    if 'X' in corners.columns and 'Y' in corners.columns:
        cx = corners['X'].tolist()
        cy = corners['Y'].tolist()
    else:
        tel_d = np.asarray(telemetry_df['Distance'].tolist())
        tel_x = np.asarray(telemetry_df['X'].tolist())
        tel_y = np.asarray(telemetry_df['Y'].tolist())
        cx, cy = [], []
        for d in dists:
            idx = int(np.argmin(np.abs(tel_d - d)))
            cx.append(float(tel_x[idx]))
            cy.append(float(tel_y[idx]))

    return dict(CornerNumber=nums, Distance=dists, X=cx, Y=cy)


# ── Main endpoint ────────────────────────────────────────────────
@analysis_bp.route('/api/analysis')
def api_analysis():
    """
    GET /api/analysis?year=&gp=&session=&driver=&lap=
    """
    year         = request.args.get('year', '2024')
    gp           = request.args.get('gp', 'Bahrain')
    session_name = request.args.get('session', 'Race')
    driver_code  = request.args.get('driver', 'VER')
    lap_num      = request.args.get('lap', 'fastest')

    try:
        import torch
        # Deferred import to avoid circular dependency
        from api_server import _load_session

        # 1 ── Load session & pick lap ──
        session = _load_session(year, gp, session_name)
        driver_laps = session.laps.pick_drivers(driver_code)

        if lap_num == 'fastest':
            lap = driver_laps.pick_fastest()
        else:
            lap_num_int = int(lap_num)
            matched = driver_laps[driver_laps['LapNumber'] == lap_num_int]
            if matched.empty:
                return jsonify({'error': f'Lap {lap_num} not found'}), 404
            lap = matched.iloc[0]

        # 2 ── Get full telemetry + circuit info ──
        telemetry = lap.get_telemetry()
        if telemetry.empty or len(telemetry) < 20:
            return jsonify({'error': 'Insufficient telemetry data'}), 400

        circuit_info = session.get_circuit_info()

        # 3 ── Convert data formats ──
        tel_json     = _fastf1_to_tel_json(telemetry)
        corners_json = _circuit_info_to_corners(circuit_info, telemetry)

        # 4 ── Write temp files & run CurveDetector pipeline ──
        with tempfile.TemporaryDirectory() as tmpdir:
            tel_path = os.path.join(tmpdir, '1_tel.json')
            cor_path = os.path.join(tmpdir, 'corners.json')
            with open(tel_path, 'w') as f:
                json.dump(tel_json, f)
            with open(cor_path, 'w') as f:
                json.dump(corners_json, f)

            # 5 ── Detect + normalise curves ──
            from src.analysis.dataset_normalization import normalize_telemetry_json
            normalized_curves = normalize_telemetry_json(
                tel_path, cor_path, DATASET_PATH
            )

        # 6 ── Load cached model resources ──
        res   = _get_analysis_resources()
        model = res['model']
        kmeans = res['kmeans']

        # 7 ── Classify each curve ──
        corners_result = []
        cluster_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        with torch.no_grad():
            for item in normalized_curves:
                curve = item['curve']
                fv    = item['normalized']

                inp    = torch.tensor(fv, dtype=torch.float32).unsqueeze(0)
                latent = model.get_latent(inp).squeeze(0).numpy()
                cid    = int(kmeans.predict(latent.reshape(1, -1))[0])
                cluster_counts[cid] += 1

                corners_result.append({
                    'corner_id':           int(curve.corner_id),
                    'cluster_id':          cid,
                    'cluster_name':        CLUSTER_NAMES.get(cid, f'Cluster {cid}'),
                    'pushing_score':       round(float(curve.pushing_score()), 1),
                    'driving_mode':        curve.get_driving_mode(),
                    'avg_speed':           round(float(curve.avg_speed()), 1),
                    'max_speed':           round(float(curve.max_speed()), 1),
                    'braking_aggression':  round(float(curve.braking_aggression()), 1),
                    'brake_pct':           round(float(curve.brake_time_percent()), 1),
                    'throttle_pct':        round(float(curve.full_throttle_percent()), 1),
                    'compound':            str(curve.compound),
                    'smoothness':          round(float(curve.smoothness_index()), 3),
                    'trajectory': {
                        'x': [round(float(v), 1) for v in curve.x],
                        'y': [round(float(v), 1) for v in curve.y],
                    },
                })

        # 8 ── Downsample track outline (max 1200 pts) ──
        tx = tel_json['tel']['x']
        ty = tel_json['tel']['y']
        if len(tx) > 1200:
            step = max(1, len(tx) // 1200)
            tx = tx[::step]
            ty = ty[::step]
        track_x = [round(float(v), 1) for v in tx]
        track_y = [round(float(v), 1) for v in ty]

        # 9 ── Corner marker positions ──
        corner_positions = [
            {'corner_id': int(n),
             'x': round(float(corners_json['X'][i]), 1),
             'y': round(float(corners_json['Y'][i]), 1)}
            for i, n in enumerate(corners_json['CornerNumber'])
        ]

        dominant = max(cluster_counts, key=cluster_counts.get) \
            if corners_result else 0

        return jsonify({
            'track':               {'x': track_x, 'y': track_y},
            'corner_positions':    corner_positions,
            'corners':             corners_result,
            'cluster_summary':     cluster_counts,
            'dominant_cluster':    dominant,
            'dominant_name':       CLUSTER_NAMES.get(dominant, '?'),
            'total_detected':      len(corners_result),
            'total_on_track':      len(corners_json['CornerNumber']),
            'cluster_names':       CLUSTER_NAMES,
            'cluster_colors':      CLUSTER_COLORS,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
