import os
import json
import traceback

import fastf1
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── FastF1 cache setup ──────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'fastf1_cache')
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

# ── Flask app ────────────────────────────────────────────────────
# Explicitly set static_folder to current directory ('.') and url_path to root ('')
# This ensures Gunicorn serves style.css and script.js correctly from the same folder.
app = Flask(__name__, static_url_path='', static_folder='.')
CORS(app)  # allow cross-origin from file:// or any dev server

# In-memory cache for loaded sessions to avoid re-downloading
_session_cache = {}


def _get_session_key(year, gp, session_name):
    return f"{year}-{gp}-{session_name}"


def _load_session(year, gp, session_name):
    """Load a FastF1 session, using in-memory cache."""
    key = _get_session_key(int(year), gp, session_name)
    if key in _session_cache:
        return _session_cache[key]

    session = fastf1.get_session(int(year), gp, session_name)
    session.load()
    _session_cache[key] = session
    return session


# ─────────────────────────────────────────────────────────────────
#  GET /api/schedule?year=2024
#  Returns list of GP event names for the given year
# ─────────────────────────────────────────────────────────────────
@app.route('/api/schedule')
def api_schedule():
    year = request.args.get('year', '2024')
    try:
        schedule = fastf1.get_event_schedule(int(year), include_testing=False)
        events = []
        for _, row in schedule.iterrows():
            events.append({
                'round': int(row['RoundNumber']),
                'name': row['EventName'],
                'country': row['Country'],
                'location': row['Location'],
            })
        return jsonify(events)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────────
#  GET /api/drivers?year=2024&gp=Bahrain&session=Race
#  Returns driver list with team info and colors
# ─────────────────────────────────────────────────────────────────
@app.route('/api/drivers')
def api_drivers():
    year = request.args.get('year', '2024')
    gp = request.args.get('gp', 'Bahrain')
    session_name = request.args.get('session', 'Race')

    try:
        session = _load_session(year, gp, session_name)
        results = session.results

        drivers = []
        for _, row in results.iterrows():
            # Get team color via fastf1.plotting if available
            team_color = '#888888'
            try:
                tc = row.get('TeamColor', '')
                if tc:
                    team_color = f'#{tc}'
            except Exception:
                pass

            drivers.append({
                'code': row.get('Abbreviation', ''),
                'name': row.get('FullName', f"{row.get('FirstName', '')} {row.get('LastName', '')}"),
                'team': row.get('TeamName', 'Unknown'),
                'color': team_color,
                'number': int(row.get('DriverNumber', 0)) if row.get('DriverNumber') else 0,
            })

        return jsonify(drivers)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────────
#  GET /api/laps?year=2024&gp=Bahrain&session=Race&driver=VER
#  Returns lap-by-lap times for a specific driver
# ─────────────────────────────────────────────────────────────────
@app.route('/api/laps')
def api_laps():
    year = request.args.get('year', '2024')
    gp = request.args.get('gp', 'Bahrain')
    session_name = request.args.get('session', 'Race')
    driver_code = request.args.get('driver', 'VER')

    try:
        session = _load_session(year, gp, session_name)
        driver_laps = session.laps.pick_drivers(driver_code)

        laps = []
        for _, lap in driver_laps.iterrows():
            lap_time = lap.get('LapTime')
            lap_seconds = lap_time.total_seconds() if lap_time and hasattr(lap_time, 'total_seconds') else None

            # Skip laps without valid times
            if lap_seconds is None or lap_seconds <= 0 or lap_seconds > 300:
                continue

            compound = ''
            try:
                compound = str(lap.get('Compound', ''))
                if compound == 'nan' or compound == 'None':
                    compound = ''
            except Exception:
                pass

            laps.append({
                'lap': int(lap.get('LapNumber', 0)),
                'time': round(lap_seconds, 3),
                'sector1': round(lap.get('Sector1Time').total_seconds(), 3) if hasattr(lap.get('Sector1Time', None), 'total_seconds') else None,
                'sector2': round(lap.get('Sector2Time').total_seconds(), 3) if hasattr(lap.get('Sector2Time', None), 'total_seconds') else None,
                'sector3': round(lap.get('Sector3Time').total_seconds(), 3) if hasattr(lap.get('Sector3Time', None), 'total_seconds') else None,
                'compound': compound,
                'isPit': bool(lap.get('PitInTime') is not None and str(lap.get('PitInTime')) != 'NaT'),
            })

        return jsonify(laps)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────────
#  GET /api/telemetry?year=2024&gp=Bahrain&session=Race
#                    &driver=VER&lap=5
#  Returns per-lap telemetry (Speed, Throttle, Brake, Gear, Distance)
# ─────────────────────────────────────────────────────────────────
@app.route('/api/telemetry')
def api_telemetry():
    year = request.args.get('year', '2024')
    gp = request.args.get('gp', 'Bahrain')
    session_name = request.args.get('session', 'Race')
    driver_code = request.args.get('driver', 'VER')
    lap_num = request.args.get('lap', 'fastest')

    try:
        session = _load_session(year, gp, session_name)
        driver_laps = session.laps.pick_drivers(driver_code)

        if lap_num == 'fastest':
            lap = driver_laps.pick_fastest()
        else:
            lap_num = int(lap_num)
            matched = driver_laps[driver_laps['LapNumber'] == lap_num]
            if matched.empty:
                return jsonify({'error': f'Lap {lap_num} not found for {driver_code}'}), 404
            lap = matched.iloc[0]

        telemetry = lap.get_telemetry()

        # Sample down if too many points (keep ≤ 500 points)
        if len(telemetry) > 500:
            step = max(1, len(telemetry) // 500)
            telemetry = telemetry.iloc[::step].reset_index(drop=True)

        data = []
        for _, row in telemetry.iterrows():
            data.append({
                'distance': round(float(row.get('Distance', 0)), 1),
                'speed': round(float(row.get('Speed', 0)), 1),
                'throttle': round(float(row.get('Throttle', 0)), 1),
                'brake': int(bool(row.get('Brake', 0))),
                'gear': int(row.get('nGear', 0)),
                'rpm': int(row.get('RPM', 0)) if row.get('RPM') else 0,
            })

        return jsonify(data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────────
#  Static files fallback (serve the site itself)
# ─────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return app.send_static_file('index.html')


if __name__ == '__main__':
    print("🏎️  RacingDNA API Server starting on http://localhost:5050")
    print("   FastF1 cache:", os.path.abspath(CACHE_DIR))
    app.static_folder = os.path.dirname(__file__)
    app.run(host='0.0.0.0', port=5050, debug=True)
