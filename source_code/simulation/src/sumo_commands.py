import datetime
import random as rd
import shutil

import numpy as np
import pandas as pd
import traci

import config as _cfg_sc
from helper import printv

_SEED = 42
_RNG_SPAWN = rd.Random(_SEED)
_RNG_REROUTE = rd.Random(_SEED + 1)
_NP_RNG_REROUTE = np.random.default_rng(_SEED + 2)


def set_seeds(seed):
    """Seed the three independent streams. The SUMO seed is passed to the binary in startSumo()."""
    global _SEED, _RNG_SPAWN, _RNG_REROUTE, _NP_RNG_REROUTE
    _SEED = seed
    _RNG_SPAWN = rd.Random(seed)
    _RNG_REROUTE = rd.Random(seed + 1)
    _NP_RNG_REROUTE = np.random.default_rng(seed + 2)


def count_non_workers_non_students_with_car(people_coordinates_on_grid):
    """
    Count the number of people who are neither employed nor in education and have a car.

    Args:
        people_coordinates_on_grid (dict): Dictionary containing people data with coordinates as keys.

    Returns:
        int: Number of people who are neither employed nor in education and have a car.
    """
    count = 0
    for data in people_coordinates_on_grid.values():
        for person in data['people']:
            is_not_employed = not person.get('position_in_bus')
            is_not_student = not person.get('position_in_edu')
            has_car = person.get('has_car_and_licence') == 'True'
            if is_not_employed and is_not_student and has_car:
                count += 1
    return count


# Share of a full-time position per `level_of_employment` category
EMPLOYMENT_WEIGHTS = {'0': 0, '1-39': 0.2, '40-79': 0.6, '80-100': 0.9}


def count_workers_with_cars_adjusted(people_coordinates_on_grid):
    """
    Count the workers with cars (weighted by their level of employment) and the whole population with cars.

    Args:
        people_coordinates_on_grid (dict): Dictionary containing people data with coordinates as keys.
    Returns:
        tuple: Number of workers with cars (full-time equivalents) and total population with cars.
    """
    workers_with_cars = 0
    population_with_car = 0

    for data in people_coordinates_on_grid.values():
        for person in data['people']:
            if person.get('has_car_and_licence') != 'True':
                continue
            population_with_car += 1
            if person.get('position_in_bus'):   # employed
                # persons with an unknown employment level are skipped
                workers_with_cars += EMPLOYMENT_WEIGHTS.get(person.get('level_of_employment'), 0)

    return int(workers_with_cars), int(population_with_car)


def startSumo(zoom, x_off, y_off, dt=1.0, visual_interface=True, verbose=True, seed=42):
    """
    Start the SUMO simulation with the specified parameters.
    Args:
        zoom (float): Zoom level for the SUMO GUI.
        x_off (float): X offset for the SUMO GUI.
        y_off (float): Y offset for the SUMO GUI.
        dt (float): Simulation time step.
        visual_interface (bool): Whether to use the SUMO GUI.
        verbose (bool): If more information should be printed.
        seed (int): SUMO random seed (--seed).

    Returns:
        None
    """
    printv("Start function startSumo", verbose=verbose, color="blue", decorate=True)

    sumoBinary = "sumo"
    if visual_interface:
        if shutil.which("sumo-gui"):
            sumoBinary = "sumo-gui"
        else:
            printv("WARNING: Could not find sumo-gui, using regular sumo", verbose=verbose, color="yellow")
            visual_interface = False

    printv(f"Using SUMO binary: {sumoBinary}", verbose=verbose)

    sumoCmd = [sumoBinary, "-c", _cfg_sc.SUMO_FILE, "--start", "--quit-on-end",
               "--step-length", str(dt), "--seed", str(seed)]

    # unique connection label from the wall-clock time
    unique = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    printv(f"Opening connection test_label_{unique}", verbose=verbose)

    try:
        traci.start(sumoCmd, label=f"test_label_{unique}")
    except Exception as e:
        printv(f"Error starting SUMO: {e}", verbose=verbose, color="red")
        raise

    # Only set GUI parameters if using the visual interface
    if visual_interface:
        try:
            traci.gui.setZoom("View #0", zoom)
            traci.gui.setOffset("View #0", x_off, y_off)
        except Exception as e:
            printv(f"Warning: Could not set GUI parameters: {e}", verbose=verbose, color="yellow")

    if verbose:
        print("Started SUMO")
        
    precompute_suitable_edges()
    create_delivery_vehicle_type()


def stopSumo(verbose=False):
    """
    End sumo simulation.

    Args:
        verbose (bool) : Tells if printed output
    """

    printv("Start function stopSumo", verbose=verbose,
           color="blue", decorate=True)

    traci.close()


# Loaded lazily on first call so the village/date override in run_sim.py
# has time to update config before the file is read.
_speed_data = None

def update_edge_speeds(time_step):
    """
    Update edge speeds based on the current simulation time. Reads the per-second exit speeds from the CSV written by route_matching/speed_regulation.py.

    Args:
        time_step (float): Current simulation time step.

    Returns:
        None
    """
    global _speed_data
    if _speed_data is None:
        _speed_data = pd.read_csv(_cfg_sc.SPEED_FILE)
        # -1.0 = no detector data yet. Back-fill the leading gap with the first
        # observed speed so the exit lanes are forced from t = 0, not only from the
        # first data point (1.1-6.6 h into the day). Columns with no data stay -1.
        for col in _speed_data.columns:
            if col == 'time':
                continue
            series = _speed_data[col].where(_speed_data[col] != -1.0)
            _speed_data[col] = series.bfill().fillna(-1.0)

    # Get the speeds for the current time step (if available)
    if int(time_step) < len(_speed_data):
        current_speeds = _speed_data.iloc[int(time_step)]
        
        # Process each edge in the CSV
        for edge in current_speeds.keys():
            if edge == 'time':  # Skip the time column
                continue
                
            speed = current_speeds[edge]
            
            # Apply speed only if not -1.0
            if speed != -1.0:
                try:
                    traci.edge.setMaxSpeed(edge, float(speed))
                except traci.TraCIException as e:
                    print(f"Error setting speed for edge {edge}: {e}")


def reroute_vehicles_to_avoid_traffic(reroute_percentage, vehicle_ids):
    """
    Reroutes a given percentage of vehicles to avoid traffic.
    
    Args:
        reroute_percentage (float): Percentage of vehicles to reroute (0.0-1.0)
        vehicle_ids (list): List of vehicle IDs to consider for rerouting.

    Returns:
        None
    """
    expected = len(vehicle_ids) * reroute_percentage
    # stochastic rounding: the expected share is met on average even with few vehicles
    num_to_reroute = int(expected) + (1 if _RNG_REROUTE.random() < expected - int(expected) else 0)

    if num_to_reroute == 0:
        return

    # Randomly select vehicles to reroute
    vehicles_to_reroute = _NP_RNG_REROUTE.choice(vehicle_ids,
                                          size=num_to_reroute,
                                          replace=False)

    for veh_id in vehicles_to_reroute:
        # Set the mode first: in default mode the reroute uses static travel times
        # and is a no-op. 1 = ROUTING_MODE_AGGREGATED (smoothed current travel times).
        traci.vehicle.setRoutingMode(veh_id, 1)
        traci.vehicle.rerouteTraveltime(veh_id, currentTravelTimes=True)


# Precompute suitable edges for delivery vehicles
# This is a global variable to store the suitable edges
SUITABLE_EDGES = []
def precompute_suitable_edges():
    """
    Precompute suitable edges for delivery vehicles to improve performance
    """
    global SUITABLE_EDGES
        
    # Get all edges that can be used as start points
    edges = traci.edge.getIDList()
    suitable_edges = []
    
    for edge in edges:
        if edge.startswith(':'):  # Skip internal/junction edges
            continue
            
        try:
            # Check if edge has lanes
            lane_count = traci.edge.getLaneNumber(edge)
            if lane_count <= 0:
                continue
                
            # An edge is eligible if ANY of its lanes is usable by private cars /
            # delivery vehicles and is below the speed threshold.
            edge_valid = False
            for lane_idx in range(lane_count):
                lane_id = f"{edge}_{lane_idx}"
                allowed = traci.lane.getAllowed(lane_id)
                lane_speed = traci.lane.getMaxSpeed(lane_id)
                delivery_allowed = (not allowed) or (('delivery' in allowed) and ('pedestrian' not in allowed)) or ('passenger' in allowed)
                if delivery_allowed and lane_speed < 17:   # < 17 m/s (~61 km/h): village roads incl. 50 km/h main road
                    edge_valid = True
                    break

            # Check speed limit (avoid highways)
            if edge_valid:
                suitable_edges.append(edge)
                
        except Exception as e:
            # Skip edges with errors
            print(f"Error processing edge {edge}: {e}")
            continue
    
    SUITABLE_EDGES = suitable_edges
    print(f"Precomputed {len(SUITABLE_EDGES)} suitable edges for delivery vehicles (speed limit < 17 m/s)")


# Counters for the whole run; printed by run_sim.py at the end.
SPAWN_STATS = {"requested": 0, "spawned": 0, "failed": 0}


def add_random_delivery_vehicle():
    """
    Add a single random local vehicle (has access to housing areas) to the simulation.
    Returns True on success. A failed attempt (no valid route after 5 tries) is
    counted and logged; unexpected (non-TraCI) errors are not caught.
    """
    SPAWN_STATS["requested"] += 1

    if len(SUITABLE_EDGES) < 2:
        SPAWN_STATS["failed"] += 1
        print("[spawn FAILED] fewer than 2 suitable edges available")
        return False

    rng = rd.Random(f"{_SEED}:{SPAWN_STATS['requested']}")   # own stream per spawn request
    last_reason = "no attempt"
    for attempt in range(5):
        from_edge = rng.choice(SUITABLE_EDGES)
        to_edge = rng.choice([e for e in SUITABLE_EDGES if e != from_edge])
        try:
            route = traci.simulation.findRoute(from_edge, to_edge, vType="resident_type", routingMode=1)
            if len(route.edges) <= 1 or route.cost >= 1000000:   # high cost indicates invalid route
                last_reason = f"invalid route {from_edge} -> {to_edge}"
                continue

            veh_id = f"resident_{int(traci.simulation.getTime())}_{rng.randint(0, 100000)}"
            route_id = f"route_{veh_id}"
            traci.route.add(route_id, route.edges)
            traci.vehicle.add(veh_id, route_id, typeID="resident_type", departSpeed="0")
            traci.vehicle.setColor(veh_id, (255, 0, 0, 255))   # red color
            traci.vehicle.setSpeedFactor(veh_id, rng.uniform(0.8, 1.0))
            SPAWN_STATS["spawned"] += 1
            return True
        except traci.exceptions.TraCIException as e:
            last_reason = f"TraCI error {from_edge} -> {to_edge}: {e}"

    SPAWN_STATS["failed"] += 1
    print(f"[spawn FAILED] t={traci.simulation.getTime():.0f}s after 5 attempts: {last_reason}")
    return False


def add_time_dependent_traffic(weekday, curr_time, dt, population_with_car, working_population, inactive_population):
    """
    Add time-dependent random traffic to the simulation
    
    Args:
        weekday (bool): false if weekend, true if weekday
        curr_time (float): Current simulation time in seconds
        dt (float): Simulation time step
        population_with_car (int): How many people lie in the area with cars
        working_population(int): How many people are working with car (adjusted to the amount of work)
        inactive_population (int): People that are not working and not a student but have a car

    Returns:
        None
    """
    # Convert current time to hours (assuming curr_time is in seconds)
    curr_hour = (curr_time / 3600.0) % 24
    
    if weekday==True:
        if 0 <= curr_hour < 5:
            target_vehicles = population_with_car*0.001  # 0.1% of people at night per hour
        elif 5<= curr_hour < 9:
            target_vehicles = working_population*(1/4.) # working population with car, go to work over 4 hours
        elif 9 <= curr_hour < 11:
            target_vehicles = inactive_population*0.2*(1/2.)  # 20% of people that stay at home, over 2 hours do something in the morning
        elif 11 <= curr_hour < 13:
            target_vehicles = working_population*0.5*(1/2.)  # 50% of workers go home for lunch, over two hours
        elif 13 <= curr_hour < 16:
            target_vehicles = inactive_population*0.15*(1/3.)   # 15 % drive around at afternoon
        elif 16 <= curr_hour < 19:
            target_vehicles = working_population*(1/3.)  # working population with car, come home from work over 3 hours
        elif 19 <= curr_hour < 21:
            target_vehicles = population_with_car*0.1*(1/2.)  # 10% of people do something in the evening, over 2 hours
        else:  # 21-24
            target_vehicles = population_with_car*0.001 # 0.1% of people at night per hour
    else:
        # Weekend traffic pattern
        if 0 <= curr_hour < 5:
            target_vehicles = population_with_car*0.001 # 0.1% of people at night per hour
        elif 5<= curr_hour < 9:
            target_vehicles = population_with_car*0.05*(1/4.) # 5% of people do something in the morning, over 4 hours
        elif 9 <= curr_hour < 18:
            target_vehicles = population_with_car*0.25*(1/9.) # 25% of people do something during the day (spread over 9 hours) -> assuming that families use one car
        elif 18 <= curr_hour < 21:
            target_vehicles = population_with_car*0.05*(1/3.) # 5% of people do something in the evening
        else:  # 21-24
            target_vehicles = population_with_car*0.001 # 0.1% of people at night per hour
            
    # target_vehicles is the expected number of spawns per hour (kept fractional, no minimum)
    expected_vehicles = target_vehicles / 3600.0 * dt   # expected spawns in this step

    # whole part always spawns, the fractional part spawns with that probability
    n_spawn = int(expected_vehicles) + (1 if _RNG_SPAWN.random() < expected_vehicles - int(expected_vehicles) else 0)
    for _ in range(n_spawn):
        add_random_delivery_vehicle()


def create_delivery_vehicle_type():
    """
    Create a custom vehicle type for delivery vehicles with specific permissions.
    """
    try:
        # Create a custom vehicle type with delivery permissions
        traci.vehicletype.copy("DEFAULT_VEHTYPE", "resident_type")
        
        # IMPORTANT: Set vehicle class to "delivery" - this is critical for route access in housing areas
        traci.vehicletype.setVehicleClass("resident_type", "delivery")
        
        traci.vehicletype.setColor("resident_type", (255, 0, 0, 255))  # Red color
        
        # Make sure it's properly registered by checking available types
        available_types = traci.vehicletype.getIDList()
        if "resident_type" in available_types:
            print("Successfully created custom delivery vehicle type: resident_type")
        else:
            print("Failed to create resident_type properly")
    except traci.exceptions.TraCIException as e:
        print(f"Error creating custom vehicle type: {e}")