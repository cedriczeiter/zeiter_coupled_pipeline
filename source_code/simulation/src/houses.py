import os

import pandas as pd
import requests
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

from helper import convert_sumo_coordinates_to_lat_lon, printv
import config as _cfg

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# OSM `building=*` values that count as a place where people live
# (besides the usual housing names, OSM often has plain `building=yes`)
RESIDENTIAL_BUILDING_TAGS = {
    'house', 'residential', 'detached', 'semidetached_house', 'terrace', 'apartments', 'hotel',
    'farm_auxiliary', 'yes', 'farm', 'bungalow', 'cabin', 'annexe', 'dormitory', 'static_caravan',
}

EMPTY_RECORD_FIELDS = ('place_id', 'osm_type', 'osm_id', 'house_number', 'road', 'polygon',
                       'village', 'state', 'postcode', 'bounding_box')


def create_geolocator(user_agent="bachelor_thesis_eth", delay=1.0, verbose=False):
    """
    Create a rate-limited reverse geocode function.

    Args:
        user_agent(str): Name under which the API will see me
        delay(float): In seconds, delay of api-calls, capped at 3.0 s
        verbose(bool): If more stuff should be printed

    Returns:
        function: A rate-limited reverse geocoding function.
    """
    printv("Start function create_geolocator", verbose=verbose, color="blue")

    delay = min(delay, 3.0)

    geolocator = Nominatim(user_agent=user_agent)
    return RateLimiter(geolocator.reverse, min_delay_seconds=delay)


def fetch_osm_building(osm_type, osm_id):
    """
    Ask the Overpass API for an OSM object. Returns (polygon, building_type): the WKT polygon
    (lon lat) of the first way, and the building tag, or (None, None) if the object is not a
    residential building (or Overpass does not answer).
    """
    if osm_type not in ("node", "way", "relation"):
        return None, None
    query = f"[out:json]; {osm_type}({osm_id}); out body; >; out skel qt;"
    response = requests.post(OVERPASS_URL, data=query, timeout=120)
    if response.status_code != 200:
        return None, None
    elements = response.json()['elements']

    building_type = next((e['tags']['building'] for e in elements
                          if e['type'] == 'way' and e.get('tags', {}).get('building') in RESIDENTIAL_BUILDING_TAGS),
                         None)
    if building_type is None:
        return None, None

    nodes = {e['id']: (e['lon'], e['lat']) for e in elements if e['type'] == 'node'}
    for e in elements:
        if e['type'] == 'way' and 'nodes' in e:
            coords = [nodes[node_id] for node_id in e['nodes'] if node_id in nodes]
            if coords:
                return "POLYGON((" + ", ".join(f"{lon} {lat}" for lon, lat in coords) + "))", building_type
    return None, building_type


def get_building_info(lat, lon, geocode_func, verbose=False):
    """
    Query Nominatim reverse geocoder for a building at (lat, lon) and try to fetch its polygon.

    Args:
        lat (float): Latitude in degrees.
        lon (float): Longitude in degrees.
        geocode_func (function): Geolocator function for reverse calls
        verbose(bool): If more stuff should be printed

    Returns:
        dictionary with the address data, the OSM polygon and an `is_house` flag; all fields
        are None if the lookup failed
    """
    printv("Start function get_building_info", verbose=verbose, color="blue")

    record = {'lat': lat, 'lon': lon, **dict.fromkeys(EMPTY_RECORD_FIELDS), 'is_house': False}
    try:
        location = geocode_func((lat, lon), exactly_one=True)
        if not location:
            return record
        address = location.raw.get('address', {})
        record.update(
            place_id=location.raw.get('place_id'),
            osm_type=location.raw.get('osm_type'),
            osm_id=location.raw.get('osm_id'),
            house_number=address.get('house_number'),
            road=address.get('road'),
            village=address.get('village'),
            state=address.get('state'),
            postcode=address.get('postcode'),
            bounding_box=location.raw.get('boundingbox'),
        )
    except Exception as e:
        printv(f"[ERROR] ({lat}, {lon}): {e}", verbose=verbose, color="red")
        return record

    if record['osm_id'] and record['osm_type']:
        try:
            record['polygon'], building_type = fetch_osm_building(record['osm_type'], record['osm_id'])
            record['is_house'] = building_type is not None
        except Exception as e:
            printv(f"[WARNING] Failed to fetch OSM polygon: {e}", verbose=verbose, color="red")
    return record


def get_house_polygons(list_coords, netfile, recalculate=False, verbose=True, output_dir="../output/temp"):
    """
    Convert multiple SUMO coordinates to lat/lon and retrieve building info from Nominatim API.

    Note:
        This is not perfect. Some API-calls dont find the houses but roads instead and will just be discarded

    Args:
        list_coords (list of (x, y)): SUMO coordinates.
        netfile (str): Path to SUMO .net.xml file.
        recalculate (bool): If True, force re-fetching data.
        verbose (bool): Print extra info.
        output_dir (str): Directory to store result CSV.

    Returns:
        pd.DataFrame: DataFrame with lat, lon, bounding_box and Polygon shape for all given coords.
    """
    printv("Start function get_house_polygons",
           verbose=verbose, color="blue", decorate=True)

    if len(list_coords) == 0:
        printv("No coordinates found in the people data get_house_polygons. Is this expected? Program continues assuming this is correct", True, "red")
        return pd.DataFrame()

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        f"house_polygons_{_cfg.GRID_LEFT}_{_cfg.GRID_RIGHT}_{_cfg.GRID_BOTTOM}_{_cfg.GRID_TOP}.csv"
    )

    if not recalculate and os.path.exists(output_file):
        printv(f"Loading existing data from {output_file}", verbose=verbose)
        return pd.read_csv(output_file)

    printv("Start converting latlon coords", verbose=verbose)
    latlon_coords = [convert_sumo_coordinates_to_lat_lon(x, y, netfile) for x, y in list_coords]

    if verbose:
        est_time = len(latlon_coords)
        printv(f"Estimated time: {est_time}s = {est_time/60:.1f} min = {est_time/3600:.2f} h", verbose=verbose)

    geocode = create_geolocator()

    printv("Start fetching information", verbose=verbose)
    df = pd.DataFrame([get_building_info(lat, lon, geocode) for lat, lon in latlon_coords])

    # only keep the buildings (OSM "way"); other types are roads or API errors
    house_polygons = df[(df['osm_type'] == 'way') & df['is_house']]

    # one row per building (several residents can share a building)
    house_polygons = house_polygons.drop_duplicates(subset=['osm_id'])
    house_polygons = house_polygons[['lat', 'lon', 'bounding_box', 'polygon']]

    house_polygons.to_csv(output_file, index=False)
    printv(f"Saved data to {output_file}", verbose=verbose)

    return house_polygons
