conda activate newspyder # Activate the conda environment -> change to your environment name

python3 parse_routes.py
python3 from_route_to_trip.py


printf "Start duarouter Erstfeld \n"
duarouter -n ../simulation/villages/Erstfeld/erstfeld_osm.net.xml --route-files output/reconstructed_trips_erstfeld.trips.xml -a data/vehicle_types_new.xml -o output/20_11_routes_erstfeld.rou.xml

printf "Start duarouter Goeschenen \n"
duarouter -n ../simulation/villages/Goeschenen/goeschenen_osm.net.xml --route-files output/reconstructed_trips_goeschenen.trips.xml -a data/vehicle_types_new.xml -o output/20_11_routes_goeschenen.rou.xml

printf "Start duarouter Gurtnellen \n"
duarouter -n ../simulation/villages/Gurtnellen/gurtnellen_osm.net.xml --route-files output/reconstructed_trips_gurtnellen.trips.xml -a data/vehicle_types_new.xml -o output/20_11_routes_gurtnellen.rou.xml

printf "Start duarouter Schattdorf \n"
duarouter -n ../simulation/villages/Schattdorf/schattdorf_osm.net.xml --route-files output/reconstructed_trips_schattdorf.trips.xml -a data/vehicle_types_new.xml -o output/20_11_routes_schattdorf.rou.xml

printf "Start duarouter Silenen \n"
duarouter -n ../simulation/villages/Silenen/silenen_osm.net.xml --route-files output/reconstructed_trips_silenen.trips.xml -a data/vehicle_types_new.xml -o output/20_11_routes_silenen.rou.xml

printf "Start duarouter Wassen \n"
duarouter -n ../simulation/villages/Wassen/wassen_osm.net.xml --route-files output/reconstructed_trips_wassen.trips.xml -a data/vehicle_types_new.xml -o output/20_11_routes_wassen.rou.xml

printf "calculate output speeds \n"
python3 speed_regulation.py