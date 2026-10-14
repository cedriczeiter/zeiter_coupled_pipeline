"""
Standalone re-implementation of the two pure-math helpers plot_aqi_bars.py
needs from simulation/src/process_annoyance.py and simulation/src/helper.py.

Those originals import `traci` / `sumolib` at module level (needed elsewhere in
the pipeline for live SUMO control), which a reviewer running only this
reproducibility notebook should not need to install. The functions below are
copied verbatim from the source modules -- no logic has changed.
"""


def mg_per_m3_to_ppm(molecular_weight, mg_per_m3=1):
    """Convert mg/m^3 to ppm using the molecular weight of the substance.
    Verbatim copy of simulation/src/helper.py:mg_per_m3_to_ppm.
    """
    molar_volume = 24.45  # L/mol, ideal gas at 25 C and 1 atm (EPA AQI reference conditions)
    return (mg_per_m3 * molar_volume) / molecular_weight


def calculate_aqi_for_pollutant_US_AQI(concentration, pollutant):
    """US EPA AQI breakpoint interpolation.
    Verbatim copy of simulation/src/process_annoyance.py:calculate_aqi_for_pollutant_US_AQI.
    """
    breakpoints = {
        'CO': [   # ppm, 8-h
            [0, 4.4, 0, 50],
            [4.5, 9.4, 51, 100],
            [9.5, 12.4, 101, 150],
            [12.5, 15.4, 151, 200],
            [15.5, 30.4, 201, 300],
            [30.5, 40.4, 301, 400],
            [40.5, 50.4, 401, 500]
        ],
        'NO2': [  # ppb, 1-h
            [0, 53, 0, 50],
            [54, 100, 51, 100],
            [101, 360, 101, 150],
            [361, 649, 151, 200],
            [650, 1249, 201, 300],
            [1250, 1649, 301, 400],
            [1650, 2049, 401, 500]
        ],
        'PM2.5': [  # ug/m3, 24-h; EPA 2024 revision (Good <= 9.0)
            [0, 9.0, 0, 50],
            [9.1, 35.4, 51, 100],
            [35.5, 55.4, 101, 150],
            [55.5, 125.4, 151, 200],
            [125.5, 225.4, 201, 300],
            [225.5, 325.4, 301, 500]
        ]
    }

    bps = breakpoints[pollutant]
    if concentration <= 0:
        return 0
    if concentration > bps[-1][1]:
        return 500
    for i, bp in enumerate(bps):
        if concentration <= bp[1]:
            if concentration >= bp[0]:
                return ((bp[3] - bp[2]) / (bp[1] - bp[0])) * (concentration - bp[0]) + bp[2]
            # Concentration falls in the gap between the previous band and this one
            # (EPA truncates instead): interpolate across the gap.
            prev = bps[i - 1]
            return prev[3] + (bp[2] - prev[3]) * (concentration - prev[1]) / (bp[0] - prev[1])
    return 500
