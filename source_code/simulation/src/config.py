EXPOSURE_LAYER = 1      # z-layer used for exposure / AQI / plots (1 = 0.3-1.2 m; layer 0 is the 0-0.3 m injection layer)
VERBOSE = True
FORCE_RECALCULATE = False
TIME_PER_SCREENSHOT = 3600  # seconds
SHOW_INTERFACE = False      # show SUMO GUI
REROUTING_PERIOD = 360      # seconds between rerouting passes
SEED = 42                   # default seed for SUMO, Python `random` and NumPy (override with --seed)

DATE = "07_20"          # the paper uses 07_20 and 20_11; 05_17 is an earlier demand set. Override with --date


# ------------------------------------------------------------------
# Horizontal diffusion coefficients [m²/s] and PM loss rate [1/s].
# Vertical diffusion is NOT configured here: K_z(z) follows Monin-Obukhov
# similarity theory with fixed inputs, see atmosphere.py.
# Calibrated against the Appendix B targets (background, near-road
# concentration, decay distance).
# ------------------------------------------------------------------

D_GASES = 640.0        # shared by CO and NOx (lambda = 0), calibrated on the congested window
D_PMX   = 25.0         # PM2.5 has its own D, calibrated jointly with lambda
DX_CO  = DY_CO  = D_GASES
DX_NOX = DY_NOX = D_GASES
DX_PMX = DY_PMX = D_PMX
LAMBDA_CO  = 0.0     # first-order loss of the excess over background [1/s]
LAMBDA_NOX = 0.0
LAMBDA_PMX = 0.02    # PM2.5 only: surrogate for deposition / removal

# ------------------------------------------------------------------
# Optional uniform wind, off by default. Spec string, see wind.parse_wind_spec:
#   "315:10"  or  "8-17=315:10;18=100:3"   (direction FROM in degrees : speed m/s)
# ------------------------------------------------------------------
WIND_SPEC = None

# ------------------------------------------------------------------
# Ambient background concentrations [mg/m^3] (Appendix B) -- the grid is
# initialised to these values and they act as the Dirichlet boundary condition
# at the domain edges (see emission_models.process_gas_step).
# ------------------------------------------------------------------
CO_DEFAULT_MGM3  = 0.170
NO2_DEFAULT_MGM3 = 0.0051
PM2_DEFAULT_MGM3 = 0.0055

# ------------------------------------------------------------------
# Village grids (SUMO coordinates [m], number of cells), one entry per village.
# Vertical layers: GRID_DIM_Z geometric layers, see emission_models.Z_BOUNDARIES.
# The village used when no --village is given is DEFAULT_VILLAGE; run_sim.py
# and the toy example override it through select_village().
# ------------------------------------------------------------------

GRID_DIM_Z = 6

VILLAGE_CONFIGS = {
    'erstfeld':   dict(GRID_LEFT=16850, GRID_RIGHT=18630, GRID_TOP=37720, GRID_BOTTOM=35000,
                       GRID_DIM_X=160, GRID_DIM_Y=250, VILLAGE_LARGE='Erstfeld'),
    'goeschenen': dict(GRID_LEFT=12600, GRID_RIGHT=13700, GRID_TOP=20000, GRID_BOTTOM=18200,
                       GRID_DIM_X=100, GRID_DIM_Y=180, VILLAGE_LARGE='Goeschenen'),
    'gurtnellen': dict(GRID_LEFT=15400, GRID_RIGHT=16500, GRID_TOP=27000, GRID_BOTTOM=24700,
                       GRID_DIM_X=110, GRID_DIM_Y=230, VILLAGE_LARGE='Gurtnellen'),
    'schattdorf': dict(GRID_LEFT=16900, GRID_RIGHT=19000, GRID_TOP=41500, GRID_BOTTOM=38300,
                       GRID_DIM_X=210, GRID_DIM_Y=310, VILLAGE_LARGE='Schattdorf'),
    'silenen':    dict(GRID_LEFT=18900, GRID_RIGHT=20100, GRID_TOP=34300, GRID_BOTTOM=29900,
                       GRID_DIM_X=115, GRID_DIM_Y=430, VILLAGE_LARGE='Silenen'),
    'wassen':     dict(GRID_LEFT=13500, GRID_RIGHT=15300, GRID_TOP=24500, GRID_BOTTOM=20900,
                       GRID_DIM_X=175, GRID_DIM_Y=360, VILLAGE_LARGE='Wassen'),
}

DEFAULT_VILLAGE = "goeschenen"


def select_village(village=None, date=None):
    """
    Set the module-level village / date settings (grid, file paths) of this module.

    Other modules read these through `import config as _cfg` (never `from config import X`,
    which would copy the value before the override).
    """
    g = globals()
    if village is not None:
        g['VILLAGE_NAME'] = village
        g.update(VILLAGE_CONFIGS[village])
    if date is not None:
        g['DATE'] = date
    g['NETWORK_FILE'] = f"../villages/{g['VILLAGE_LARGE']}/{g['VILLAGE_NAME']}_osm.net.xml"
    g['SUMO_FILE']    = f"../villages/{g['VILLAGE_LARGE']}/{g['DATE']}_{g['VILLAGE_NAME']}.sumocfg"
    g['SPEED_FILE']   = f"../villages/{g['VILLAGE_LARGE']}/{g['DATE']}_exit_speeds_per_second_{g['VILLAGE_NAME']}.csv"


select_village(DEFAULT_VILLAGE)
