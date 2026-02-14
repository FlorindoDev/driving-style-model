/**
 * RacingDNA – Script.js
 * Frontend for F1 telemetry visualization, powered by FastF1 backend API.
 */

// ============================================================
//  CONFIG
// ============================================================

const API_BASE = '';

const FALLBACK_DRIVERS = [
    { code: 'VER', name: 'Max Verstappen', team: 'Red Bull Racing', color: '#3671C6' },
    { code: 'NOR', name: 'Lando Norris', team: 'McLaren', color: '#FF8000' },
    { code: 'HAM', name: 'Lewis Hamilton', team: 'Mercedes', color: '#27F4D2' },
    { code: 'LEC', name: 'Charles Leclerc', team: 'Ferrari', color: '#E80020' },
    { code: 'SAI', name: 'Carlos Sainz', team: 'Ferrari', color: '#FF6666' },
];

// ============================================================
//  APP STATE
// ============================================================

const state = {
    selectedDrivers: ['VER', 'NOR'],
    selectedLap: 'fastest',
    dataSource: 'api',
    cache: {
        key: null,
        lapData: {},
        telemData: {},
        drivers: null,
    }
};

// ============================================================
//  CHART INSTANCES & CROSSHAIR STATE
// ============================================================

let lapChart, speedChart, throttleChart, brakeChart, gearChart;

// Crosshair state – stores the raw mouse X pixel relative to chart area
let crosshairPixelX = null;  // pixel position on source chart
let crosshairDataX = null;   // data value (distance) for syncing across charts

// ============================================================
//  INITIALIZATION
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    setupEventListeners();
    loadSchedule().then(() => loadSession());
});

// ============================================================
//  SCHEDULE & DRIVER LIST  (from FastF1 API)
// ============================================================

async function loadSchedule() {
    const year = document.getElementById('year-select').value;
    const raceSelect = document.getElementById('race-select');

    try {
        const resp = await fetch(`${API_BASE}/api/schedule?year=${year}`);
        if (!resp.ok) throw new Error('Schedule API failed');
        const events = await resp.json();

        raceSelect.innerHTML = '';
        events.forEach(ev => {
            const opt = document.createElement('option');
            opt.value = ev.name;
            opt.text = ev.name;
            raceSelect.add(opt);
        });
    } catch (err) {
        console.warn('Failed to load schedule from API, using fallback:', err);
        raceSelect.innerHTML = '<option value="Bahrain">Bahrain Grand Prix</option>';
    }
}

async function loadDriversFromAPI(year, gp, session) {
    try {
        const url = `${API_BASE}/api/drivers?year=${year}&gp=${encodeURIComponent(gp)}&session=${encodeURIComponent(session)}`;
        const resp = await fetch(url);
        if (!resp.ok) throw new Error('Drivers API failed');
        const drivers = await resp.json();

        if (drivers && drivers.length > 0) {
            differentiateTeamColors(drivers);
            state.cache.drivers = drivers;
        } else {
            state.cache.drivers = [...FALLBACK_DRIVERS];
        }
    } catch (err) {
        console.warn('Failed to load drivers:', err);
        state.cache.drivers = [...FALLBACK_DRIVERS];
    }

    populateDriverList();
}

function populateDriverList() {
    const list = document.getElementById('driver-list');
    list.innerHTML = '';

    const drivers = state.cache.drivers || FALLBACK_DRIVERS;

    drivers.forEach(driver => {
        const item = document.createElement('div');
        item.className = `driver-item ${state.selectedDrivers.includes(driver.code) ? 'selected' : ''}`;
        item.dataset.driver = driver.code;
        item.style.setProperty('--driver-color', driver.color);

        item.innerHTML = `
            <div class="checkbox-indicator"></div>
            <span class="color-dot"></span>
            <span class="driver-name">${driver.name}</span>
            <span class="driver-team">${driver.team}</span>
        `;

        item.addEventListener('click', () => toggleDriver(driver.code));
        list.appendChild(item);
    });
}

function toggleDriver(driverCode) {
    if (state.selectedDrivers.includes(driverCode)) {
        if (state.selectedDrivers.length > 1) {
            state.selectedDrivers = state.selectedDrivers.filter(d => d !== driverCode);
        }
    } else {
        if (state.selectedDrivers.length < 5) {
            state.selectedDrivers.push(driverCode);
        }
    }

    document.querySelectorAll('.driver-item').forEach(el => {
        el.classList.toggle('selected', state.selectedDrivers.includes(el.dataset.driver));
    });

    updateLegend();

    loadSelectedDriversData().then(() => {
        updateAllCharts();
    });
}

function setupEventListeners() {
    document.getElementById('load-data-btn').addEventListener('click', () => loadSession());

    document.getElementById('year-select').addEventListener('change', () => {
        loadSchedule();
    });

    document.getElementById('lap-select').addEventListener('change', (e) => {
        state.selectedLap = e.target.value;
        loadTelemetryForSelectedLap().then(() => {
            updateTelemetryCharts();
            updateLapInfo();
        });
    });

    // Reset zoom button
    document.getElementById('reset-zoom-btn').addEventListener('click', () => {
        [lapChart, speedChart, throttleChart, brakeChart, gearChart].forEach(c => {
            if (c) c.resetZoom();
        });
    });
}

// ============================================================
//  DATA LOADING
// ============================================================

async function loadSession() {
    const year = document.getElementById('year-select').value;
    const race = document.getElementById('race-select').value;
    const session = document.getElementById('session-select').value;
    const cacheKey = `${year}-${race}-${session}`;

    if (!race) return;
    if (state.cache.key === cacheKey) return;

    showLoading(true);
    setBtnLoading(true);

    state.cache.key = cacheKey;
    state.cache.lapData = {};
    state.cache.telemData = {};
    state.selectedLap = 'fastest';

    try {
        await loadDriversFromAPI(year, race, session);

        const driverCodes = (state.cache.drivers || []).map(d => d.code);
        state.selectedDrivers = state.selectedDrivers.filter(c => driverCodes.includes(c));
        if (state.selectedDrivers.length === 0 && driverCodes.length > 0) {
            state.selectedDrivers = driverCodes.slice(0, 2);
        }
        document.querySelectorAll('.driver-item').forEach(el => {
            el.classList.toggle('selected', state.selectedDrivers.includes(el.dataset.driver));
        });

        await loadSelectedDriversData();
        state.dataSource = 'api';
    } catch (err) {
        console.error('Session load failed:', err);
        state.dataSource = 'mock';
        loadMockData();
    }

    populateLapSelector();
    updateLegend();
    updateAllCharts();
    updateSessionInfo();

    showLoading(false);
    setBtnLoading(false);
}

async function loadSelectedDriversData() {
    const year = document.getElementById('year-select').value;
    const race = document.getElementById('race-select').value;
    const session = document.getElementById('session-select').value;

    const promises = state.selectedDrivers.map(async (code) => {
        if (!state.cache.lapData[code]) {
            await loadDriverLaps(year, race, session, code);
        }
        if (!state.cache.telemData[code]) {
            await loadDriverTelemetry(year, race, session, code, state.selectedLap);
        }
    });

    await Promise.all(promises);
}

async function loadDriverLaps(year, race, session, driverCode) {
    try {
        const url = `${API_BASE}/api/laps?year=${year}&gp=${encodeURIComponent(race)}&session=${encodeURIComponent(session)}&driver=${driverCode}`;
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`Laps API failed for ${driverCode}`);
        const laps = await resp.json();

        if (laps && laps.length > 0) {
            state.cache.lapData[driverCode] = laps.map(l => ({
                x: l.lap,
                y: l.time,
                driver: driverCode,
                tire: l.compound || '',
                isPit: l.isPit || false,
                sector1: l.sector1,
                sector2: l.sector2,
                sector3: l.sector3,
            }));
        } else {
            state.cache.lapData[driverCode] = generateLapData(driverCode);
        }
    } catch (err) {
        console.warn(`Lap data failed for ${driverCode}:`, err);
        state.cache.lapData[driverCode] = generateLapData(driverCode);
    }
}

async function loadDriverTelemetry(year, race, session, driverCode, lap) {
    try {
        const lapParam = lap === 'fastest' ? 'fastest' : lap;
        const url = `${API_BASE}/api/telemetry?year=${year}&gp=${encodeURIComponent(race)}&session=${encodeURIComponent(session)}&driver=${driverCode}&lap=${lapParam}`;
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`Telemetry API failed for ${driverCode}`);
        const telem = await resp.json();

        if (telem && telem.length > 0) {
            state.cache.telemData[driverCode] = telem.map(t => ({
                x: t.distance,
                speed: t.speed,
                throttle: t.throttle,
                brake: t.brake * 100,
                gear: t.gear,
                rpm: t.rpm,
            }));
        } else {
            state.cache.telemData[driverCode] = generateTelemetry(driverCode);
        }
    } catch (err) {
        console.warn(`Telemetry failed for ${driverCode}:`, err);
        state.cache.telemData[driverCode] = generateTelemetry(driverCode);
    }
}

async function loadTelemetryForSelectedLap() {
    const year = document.getElementById('year-select').value;
    const race = document.getElementById('race-select').value;
    const session = document.getElementById('session-select').value;

    state.cache.telemData = {};

    const promises = state.selectedDrivers.map(code =>
        loadDriverTelemetry(year, race, session, code, state.selectedLap)
    );
    await Promise.all(promises);
}

// ============================================================
//  MOCK DATA GENERATORS (fallback)
// ============================================================

function loadMockData() {
    state.cache.drivers = [...FALLBACK_DRIVERS];
    state.cache.lapData = {};
    state.cache.telemData = {};

    FALLBACK_DRIVERS.forEach(d => {
        state.cache.lapData[d.code] = generateLapData(d.code);
        state.cache.telemData[d.code] = generateTelemetry(d.code);
    });

    populateDriverList();
}

function generateLapData(driverCode, laps = 55) {
    const data = [];
    const driverIdx = (state.cache.drivers || FALLBACK_DRIVERS).findIndex(d => d.code === driverCode);
    let baseTime = 88 + (driverIdx >= 0 ? driverIdx * 0.08 : Math.random());
    let currentTire = 'SOFT';
    let tireAge = 0;

    for (let i = 1; i <= laps; i++) {
        tireAge++;
        if (tireAge > 14 + Math.floor(Math.random() * 6)) {
            currentTire = currentTire === 'SOFT' ? 'MEDIUM' : 'HARD';
            tireAge = 0;
            data.push({ x: i, y: baseTime + 18 + Math.random() * 4, tire: currentTire, driver: driverCode, isPit: true });
            continue;
        }
        let time = baseTime + (tireAge * 0.08) + (Math.random() * 0.6 - 0.3);
        if (currentTire === 'SOFT') time -= 0.3;
        if (currentTire === 'HARD') time += 0.2;
        data.push({ x: i, y: time, tire: currentTire, driver: driverCode });
    }
    return data;
}

function generateTelemetry(driverCode, length = 300) {
    const data = [];
    let speed = 200, throttle = 80, brake = 0, gear = 5;
    const idx = (state.cache.drivers || FALLBACK_DRIVERS).findIndex(d => d.code === driverCode);
    const seed = (idx + 1) * 7;

    for (let i = 0; i < length; i++) {
        const phase = (i + seed) % 60;
        if (phase < 35) {
            throttle = Math.min(100, 60 + phase * 1.5 + (Math.random() - 0.5) * 5);
            brake = 0;
            speed += throttle * 0.03 - 0.5;
            gear = speed > 280 ? 8 : speed > 230 ? 7 : speed > 180 ? 6 : speed > 130 ? 5 : 4;
        } else if (phase < 45) {
            throttle = 0;
            brake = Math.min(100, 40 + (phase - 35) * 8);
            speed -= brake * 0.15;
            gear = speed > 200 ? 6 : speed > 150 ? 5 : speed > 100 ? 4 : 3;
        } else {
            throttle = Math.min(60, 10 + (phase - 45) * 4);
            brake = Math.max(0, 30 - (phase - 45) * 3);
            speed += (throttle - brake) * 0.05;
            gear = speed > 160 ? 5 : speed > 120 ? 4 : 3;
        }
        speed = Math.max(60, Math.min(340, speed + (Math.random() - 0.5) * 3));
        data.push({
            x: i * 15,
            speed: Math.round(speed),
            throttle: Math.round(Math.max(0, Math.min(100, throttle))),
            brake: Math.round(Math.max(0, Math.min(100, brake))),
            gear: Math.round(gear)
        });
    }
    return data;
}

// ============================================================
//  CHARTING
// ============================================================

function initCharts() {
    Chart.defaults.color = '#7a7a95';
    Chart.defaults.borderColor = '#252540';
    Chart.defaults.font.family = "'Titillium Web', sans-serif";
    Chart.defaults.font.size = 11;

    // -------------------------------------------------------
    //  Vertical Crosshair Plugin
    //  Draws at the stored crosshairDataX value (data space).
    //  Does NOT trigger any chart.update() – purely visual overlay.
    // -------------------------------------------------------
    const verticalLinePlugin = {
        id: 'verticalLine',
        afterDraw: (chart) => {
            if (chart.config.type === 'scatter') return;
            if (crosshairDataX === null) return;

            const ctx = chart.ctx;
            const yAxis = chart.scales.y;
            const xAxis = chart.scales.x;

            const xPixel = xAxis.getPixelForValue(crosshairDataX);
            if (xPixel < xAxis.left || xPixel > xAxis.right) return;

            // Gradient line
            ctx.save();
            const grad = ctx.createLinearGradient(0, yAxis.top, 0, yAxis.bottom);
            grad.addColorStop(0, 'rgba(225, 6, 0, 0.7)');
            grad.addColorStop(0.5, 'rgba(255, 255, 255, 0.5)');
            grad.addColorStop(1, 'rgba(225, 6, 0, 0.7)');

            ctx.beginPath();
            ctx.moveTo(xPixel, yAxis.top);
            ctx.lineTo(xPixel, yAxis.bottom);
            ctx.lineWidth = 1.5;
            ctx.strokeStyle = grad;
            ctx.stroke();

            // Dots at intersection with each dataset
            chart.data.datasets.forEach((ds, dsIdx) => {
                const meta = chart.getDatasetMeta(dsIdx);
                if (meta.hidden) return;
                const elements = meta.data;
                let closest = null;
                let minDist = Infinity;
                for (let i = 0; i < elements.length; i++) {
                    const d = Math.abs(elements[i].x - xPixel);
                    if (d < minDist) { minDist = d; closest = elements[i]; }
                }
                if (closest && minDist < 20) {
                    ctx.beginPath();
                    ctx.arc(closest.x, closest.y, 4, 0, Math.PI * 2);
                    ctx.fillStyle = ds.borderColor;
                    ctx.fill();
                    ctx.strokeStyle = '#fff';
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                }
            });

            ctx.restore();
        }
    };

    Chart.register(verticalLinePlugin);

    // -------------------------------------------------------
    //  Tooltip style
    // -------------------------------------------------------
    const tooltipStyle = {
        backgroundColor: 'rgba(15, 15, 30, 0.95)',
        titleColor: '#fff',
        bodyColor: '#ccc',
        borderColor: '#e10600',
        borderWidth: 1,
        cornerRadius: 4,
        padding: 10,
        titleFont: { weight: '700', size: 12 },
        bodyFont: { size: 11 },
        displayColors: true,
        boxWidth: 8,
        boxHeight: 8,
        callbacks: {
            labelColor: function (ctx) {
                return {
                    borderColor: ctx.dataset.borderColor,
                    backgroundColor: ctx.dataset.borderColor || ctx.dataset.backgroundColor,
                    borderRadius: 2
                };
            }
        }
    };

    // -------------------------------------------------------
    //  Zoom plugin (drag only, NO wheel, double-click to reset)
    // -------------------------------------------------------
    const zoomOptions = {
        zoom: {
            wheel: { enabled: false },  // DISABLED – user request
            pinch: { enabled: true },
            drag: {
                enabled: true,
                backgroundColor: 'rgba(225, 6, 0, 0.1)',
                borderColor: 'rgba(225, 6, 0, 0.4)',
                borderWidth: 1,
            },
            mode: 'x',
            onZoomComplete: () => { } // noop
        },
        pan: {
            enabled: true,
            mode: 'x',
            modifierKey: 'ctrl'
        }
    };

    // -------------------------------------------------------
    //  Common options for telemetry line charts
    //  NOTE: NO onHover callback – crosshair is handled via
    //  raw mousemove events attached directly to canvas elements.
    // -------------------------------------------------------
    const commonLineOptions = {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
            legend: { display: false },
            tooltip: {
                ...tooltipStyle,
                enabled: false // We use external tooltip via sync
            },
            zoom: zoomOptions
        },
        scales: {
            x: { type: 'linear', display: false },
            y: { grid: { color: '#1f1f35', lineWidth: 0.5 } }
        },
        elements: { point: { radius: 0, hitRadius: 10 }, line: { borderWidth: 2 } },
        animation: { duration: 0 },
    };

    // -------------------------------------------------------
    //  1. Lap Analysis Chart (Scatter)
    // -------------------------------------------------------
    const ctxLap = document.getElementById('lapChart').getContext('2d');
    lapChart = new Chart(ctxLap, {
        type: 'scatter',
        data: { datasets: [] },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 400 },
            interaction: { mode: 'index', intersect: false },
            scales: {
                x: {
                    type: 'linear', position: 'bottom',
                    title: { display: true, text: 'Lap Number', color: '#50506a', font: { size: 10 } },
                    grid: { color: '#1f1f35', lineWidth: 0.5 }
                },
                y: {
                    title: { display: true, text: 'Lap Time (s)', color: '#50506a', font: { size: 10 } },
                    grid: { color: '#1f1f35', lineWidth: 0.5 }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    ...tooltipStyle,
                    callbacks: {
                        title: (items) => `Lap ${items[0]?.raw?.x || ''}`,
                        label: (ctx) => {
                            const p = ctx.raw;
                            const timeStr = formatLapTime(p.y);
                            const tire = p.tire ? ` · ${p.tire}` : '';
                            const pit = p.isPit ? ' ⟳ PIT' : '';
                            return ` ${p.driver}: ${timeStr}${tire}${pit}`;
                        }
                    }
                },
                zoom: zoomOptions
            },
            onClick: (e, elements) => {
                if (elements.length > 0) {
                    const idx = elements[0].index;
                    const dsIdx = elements[0].datasetIndex;
                    const point = lapChart.data.datasets[dsIdx].data[idx];
                    const lapNum = point.x;

                    const lapSelect = document.getElementById('lap-select');
                    lapSelect.value = lapNum.toString();
                    state.selectedLap = lapNum.toString();

                    loadTelemetryForSelectedLap().then(() => {
                        updateTelemetryCharts();
                        updateLapInfo();
                    });
                }
            }
        }
    });

    // -------------------------------------------------------
    //  2. Speed Chart
    // -------------------------------------------------------
    speedChart = new Chart(document.getElementById('speedChart').getContext('2d'), {
        type: 'line', data: { datasets: [] },
        options: {
            ...commonLineOptions,
            plugins: {
                ...commonLineOptions.plugins,
                tooltip: {
                    ...tooltipStyle,
                    callbacks: {
                        ...tooltipStyle.callbacks,
                        label: (ctx) => ` ${ctx.dataset.label}: ${ctx.parsed.y} km/h`
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear', display: true,
                    title: { display: true, text: 'Distance (m)', color: '#50506a', font: { size: 10 } },
                    grid: { color: '#1f1f35', lineWidth: 0.5 }
                },
                y: { grid: { color: '#1f1f35', lineWidth: 0.5 } }
            }
        }
    });

    // -------------------------------------------------------
    //  3. Throttle Chart
    // -------------------------------------------------------
    throttleChart = new Chart(document.getElementById('throttleChart').getContext('2d'), {
        type: 'line', data: { datasets: [] },
        options: {
            ...commonLineOptions,
            plugins: {
                ...commonLineOptions.plugins,
                tooltip: {
                    ...tooltipStyle,
                    callbacks: {
                        ...tooltipStyle.callbacks,
                        label: (ctx) => ` ${ctx.dataset.label}: ${ctx.parsed.y}%`
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear', display: true,
                    title: { display: true, text: 'Distance (m)', color: '#50506a', font: { size: 10 } },
                    grid: { color: '#1f1f35', lineWidth: 0.5 }
                },
                y: { ...commonLineOptions.scales.y, min: 0, max: 105 }
            }
        }
    });

    // -------------------------------------------------------
    //  4. Brake Chart
    // -------------------------------------------------------
    brakeChart = new Chart(document.getElementById('brakeChart').getContext('2d'), {
        type: 'line', data: { datasets: [] },
        options: {
            ...commonLineOptions,
            plugins: {
                ...commonLineOptions.plugins,
                tooltip: {
                    ...tooltipStyle,
                    callbacks: {
                        ...tooltipStyle.callbacks,
                        label: (ctx) => ` ${ctx.dataset.label}: ${ctx.parsed.y}%`
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear', display: true,
                    title: { display: true, text: 'Distance (m)', color: '#50506a', font: { size: 10 } },
                    grid: { color: '#1f1f35', lineWidth: 0.5 }
                },
                y: { ...commonLineOptions.scales.y, min: 0, max: 105 }
            }
        }
    });

    // -------------------------------------------------------
    //  5. Gear Chart
    // -------------------------------------------------------
    gearChart = new Chart(document.getElementById('gearChart').getContext('2d'), {
        type: 'line', data: { datasets: [] },
        options: {
            ...commonLineOptions,
            plugins: {
                ...commonLineOptions.plugins,
                tooltip: {
                    ...tooltipStyle,
                    callbacks: {
                        ...tooltipStyle.callbacks,
                        label: (ctx) => ` ${ctx.dataset.label}: Gear ${ctx.parsed.y}`
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear', display: true,
                    title: { display: true, text: 'Distance (m)', color: '#50506a', font: { size: 10 } },
                    grid: { color: '#1f1f35', lineWidth: 0.5 }
                },
                y: { ...commonLineOptions.scales.y, min: 0, max: 9, ticks: { stepSize: 1 } }
            },
            elements: { ...commonLineOptions.elements, line: { borderWidth: 2, stepped: 'before' } }
        }
    });

    // -------------------------------------------------------
    //  RAW MOUSEMOVE CROSSHAIR (No Chart.js hover system)
    //  This is the key fix for flickering. We attach native
    //  mousemove/mouseleave events to each telemetry canvas
    //  and manually draw + trigger tooltips using requestAnimationFrame.
    // -------------------------------------------------------
    setupCrosshairSync();
}

/**
 * Attach raw mousemove/mouseleave events to each telemetry chart canvas.
 * On move: convert mouse pixel X → data X value → store in crosshairDataX
 *          then requestAnimationFrame to redraw all charts & sync tooltips.
 */
function setupCrosshairSync() {
    const telemetryCharts = () => [speedChart, throttleChart, brakeChart, gearChart];
    const canvasIds = ['speedChart', 'throttleChart', 'brakeChart', 'gearChart'];

    let rafId = null;

    canvasIds.forEach(canvasId => {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return;

        canvas.addEventListener('mousemove', (evt) => {
            // Find which chart this canvas belongs to
            const chart = telemetryCharts().find(c => c && c.canvas === canvas);
            if (!chart || !chart.scales || !chart.scales.x) return;

            const rect = canvas.getBoundingClientRect();
            const mouseX = evt.clientX - rect.left;

            // Convert pixel to data value
            const xAxis = chart.scales.x;
            const dataX = xAxis.getValueForPixel(mouseX);

            // Only update if value actually changed meaningfully (reduce redraws)
            if (crosshairDataX !== null && Math.abs(crosshairDataX - dataX) < 0.5) return;

            crosshairDataX = dataX;

            // Cancel any pending frame and schedule a new one
            if (rafId) cancelAnimationFrame(rafId);
            rafId = requestAnimationFrame(() => {
                syncAllTelemetryCharts(chart);
                rafId = null;
            });
        });

        canvas.addEventListener('mouseleave', () => {
            crosshairDataX = null;
            if (rafId) cancelAnimationFrame(rafId);
            rafId = requestAnimationFrame(() => {
                clearAllTelemetryCrosshairs();
                rafId = null;
            });
        });
    });
}

/**
 * Sync crosshair + tooltips on all telemetry charts.
 * We find the nearest data index for the current crosshairDataX,
 * then activate tooltips on all charts at that index.
 */
function syncAllTelemetryCharts(sourceChart) {
    const charts = [speedChart, throttleChart, brakeChart, gearChart];

    charts.forEach(chart => {
        if (!chart || chart.data.datasets.length === 0) return;

        const ds = chart.data.datasets[0];
        if (!ds || !ds.data || ds.data.length === 0) return;

        // Binary search for nearest index by X value
        const idx = findNearestIndex(ds.data, crosshairDataX);

        // Activate tooltip elements at this index
        const activeElements = [];
        chart.data.datasets.forEach((dataset, dsIdx) => {
            const meta = chart.getDatasetMeta(dsIdx);
            if (!meta.hidden && meta.data[idx]) {
                activeElements.push({ datasetIndex: dsIdx, index: idx });
            }
        });

        chart.tooltip.setActiveElements(activeElements, { x: 0, y: 0 });
        chart.setActiveElements(activeElements);
        chart.draw(); // Direct draw, no update() – avoids triggering events
    });
}

/**
 * Clear crosshair and tooltips from all telemetry charts.
 */
function clearAllTelemetryCrosshairs() {
    const charts = [speedChart, throttleChart, brakeChart, gearChart];
    charts.forEach(chart => {
        if (!chart) return;
        chart.tooltip.setActiveElements([], { x: 0, y: 0 });
        chart.setActiveElements([]);
        chart.draw();
    });
}

/**
 * Binary search to find the data point index nearest to targetX.
 */
function findNearestIndex(data, targetX) {
    if (!data || data.length === 0) return 0;
    let lo = 0, hi = data.length - 1;
    while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (data[mid].x < targetX) lo = mid + 1;
        else hi = mid;
    }
    // Check if lo-1 is closer
    if (lo > 0 && Math.abs(data[lo - 1].x - targetX) < Math.abs(data[lo].x - targetX)) {
        return lo - 1;
    }
    return lo;
}

// ============================================================
//  CHART UPDATES
// ============================================================

function updateAllCharts() {
    updateLapChart();
    updateTelemetryCharts();
}

function updateLapChart() {
    const drivers = state.cache.drivers || FALLBACK_DRIVERS;

    const datasets = state.selectedDrivers.map(code => {
        const driver = drivers.find(d => d.code === code);
        if (!driver) return null;

        if (!state.cache.lapData[code]) {
            state.cache.lapData[code] = generateLapData(code);
        }
        const lapData = state.cache.lapData[code];

        return {
            label: driver.name,
            data: lapData,
            backgroundColor: driver.color,
            borderColor: driver.color,
            showLine: true,
            borderWidth: 2,
            tension: 0.35,
            pointRadius: 3,
            pointHoverRadius: 7,
            pointBorderWidth: 0,
            pointHoverBorderWidth: 2,
            pointHoverBorderColor: '#fff',
            fill: false,
        };
    }).filter(Boolean);

    lapChart.data.datasets = datasets;
    lapChart.update();
}

function updateTelemetryCharts() {
    const drivers = state.cache.drivers || FALLBACK_DRIVERS;
    const datasetsSpeed = [];
    const datasetsThrottle = [];
    const datasetsBrake = [];
    const datasetsGear = [];

    state.selectedDrivers.forEach(code => {
        const driver = drivers.find(d => d.code === code);
        if (!driver) return;

        if (!state.cache.telemData[code]) {
            state.cache.telemData[code] = generateTelemetry(code);
        }
        const telemData = state.cache.telemData[code];

        const commonDs = {
            label: driver.name,
            borderColor: driver.color,
            borderWidth: 2,
            tension: 0.35,
            pointRadius: 0,
            pointHoverRadius: 5,
            fill: false
        };

        datasetsSpeed.push({ ...commonDs, data: telemData.map(d => ({ x: d.x, y: d.speed })) });
        datasetsThrottle.push({ ...commonDs, data: telemData.map(d => ({ x: d.x, y: d.throttle })) });
        datasetsBrake.push({ ...commonDs, data: telemData.map(d => ({ x: d.x, y: d.brake })) });
        datasetsGear.push({
            ...commonDs,
            data: telemData.map(d => ({ x: d.x, y: d.gear })),
            stepped: 'before',
            tension: 0
        });
    });

    speedChart.data.datasets = datasetsSpeed;
    speedChart.update();

    throttleChart.data.datasets = datasetsThrottle;
    throttleChart.update();

    brakeChart.data.datasets = datasetsBrake;
    brakeChart.update();

    gearChart.data.datasets = datasetsGear;
    gearChart.update();
}

// ============================================================
//  LEGEND & LAP SELECTOR
// ============================================================

function updateLegend() {
    const legend = document.getElementById('lap-legend');
    const drivers = state.cache.drivers || FALLBACK_DRIVERS;
    legend.innerHTML = '';

    state.selectedDrivers.forEach(code => {
        const driver = drivers.find(d => d.code === code);
        if (!driver) return;
        const item = document.createElement('div');
        item.className = 'legend-item';
        item.innerHTML = `<span class="legend-dot" style="background:${driver.color}; box-shadow: 0 0 6px ${driver.color}55;"></span>${driver.code}`;
        legend.appendChild(item);
    });
}

function populateLapSelector() {
    const select = document.getElementById('lap-select');
    select.innerHTML = '<option value="fastest">Fastest Lap</option>';

    let maxLaps = 0;
    Object.values(state.cache.lapData).forEach(laps => {
        if (laps && laps.length > maxLaps) maxLaps = laps.length;
    });

    for (let i = 1; i <= maxLaps; i++) {
        const option = document.createElement('option');
        option.value = i;
        option.text = `Lap ${i}`;
        select.add(option);
    }
}

function updateLapInfo() {
    const info = document.getElementById('selected-lap-info');
    if (state.selectedLap === 'fastest') {
        info.textContent = 'Showing fastest lap telemetry';
    } else {
        info.textContent = `Showing telemetry for Lap ${state.selectedLap}`;
    }
}

// ============================================================
//  UI HELPERS
// ============================================================

function showLoading(show) {
    const overlay = document.getElementById('loading-overlay');
    overlay.style.display = show ? 'flex' : 'none';
}

function setBtnLoading(loading) {
    const btn = document.getElementById('load-data-btn');
    const text = btn.querySelector('.btn-text');
    const loader = btn.querySelector('.btn-loader');
    if (loading) {
        text.textContent = 'Loading';
        loader.style.display = 'block';
        btn.disabled = true;
    } else {
        text.textContent = 'Load Session';
        loader.style.display = 'none';
        btn.disabled = false;
    }
}

function updateSessionInfo() {
    const info = document.getElementById('session-info');
    const badge = state.dataSource === 'api'
        ? '<span class="status-badge live">LIVE DATA</span>'
        : '<span class="status-badge mock">MOCK DATA</span>';
    info.innerHTML = badge;
}

function formatLapTime(seconds) {
    if (!seconds || seconds <= 0) return '--:--.---';
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toFixed(3).padStart(6, '0')}`;
}

function differentiateTeamColors(drivers) {
    const teamCount = {};
    drivers.forEach(d => {
        const team = d.team;
        if (!teamCount[team]) teamCount[team] = [];
        teamCount[team].push(d);
    });

    Object.values(teamCount).forEach(teamDrivers => {
        if (teamDrivers.length > 1) {
            for (let i = 1; i < teamDrivers.length; i++) {
                teamDrivers[i].color = lightenColor(teamDrivers[0].color, 30 + i * 15);
            }
        }
    });
}

function lightenColor(hex, amount) {
    hex = hex.replace('#', '');
    if (hex.length === 3) hex = hex.split('').map(c => c + c).join('');
    const r = Math.min(255, parseInt(hex.substring(0, 2), 16) + amount);
    const g = Math.min(255, parseInt(hex.substring(2, 4), 16) + amount);
    const b = Math.min(255, parseInt(hex.substring(4, 6), 16) + amount);
    return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
}
