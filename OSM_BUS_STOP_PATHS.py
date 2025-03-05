import osmnx as ox
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd
import pickle
import sys
import os
from shapely.geometry import Point

###################################################################################################################

# osm_data_run är den huvudsakliga funktionen, den kräver bara ett kommun namn som det står på openstreetmap.org
# text osm_data_run('Linköpings kommun') eller osm_data_run('Alingsås kommun')
# koden kan ta ganska lång tid på sig för större kommuner med många busshållsplatser
# Koden bör användas tillsammans med en lista på stationer, koden antar att dessa stationer har koordinater i SWEREF 99 TM

# Function to get coordinates from geometry (Point or Polygon)
def get_coords(geometry):
    if geometry.geom_type == 'Point':
        return geometry.y, geometry.x
    elif geometry.geom_type == 'Polygon' or geometry.geom_type == 'MultiPolygon':
        return geometry.centroid.y, geometry.centroid.x
    return None

# Main Function to download OSM data and calculate travel times. speed är i kilometer per timme, max_distance är i meter
def osm_data_run(municipality_name, speed=5, max_distance=1000, plot_check=True,save_osm_check=True,only_download=False,output_folder=None,staion_data=None,station_columns_names = None):

    if output_folder is None:
        output_folder = os.path.dirname(__file__)

    try:
        # Download the road network and features
        G_roads = ox.graph_from_place(municipality_name, network_type="all")
        buildings = ox.features_from_place(municipality_name, tags={"building": True})
        waterways = ox.features_from_place(municipality_name, tags={"waterway": True})

        if G_roads is None:
            print('No road data, something is strange, check the name of the area in OSM, it need to be exact')
            return

        speed = speed/3.6 # From Km/h to m/s

        if staion_data is None:

            print('no station data, getting station data from OSM instead!')

            bus_stops = ox.features_from_place(municipality_name, tags={"highway": "bus_stop", "public_transport": "platform"})
            
            if bus_stops is None:
                print('No bus stop data, something is strange, check the name of the area in OSM, it need to be exact')
                return

            # Get bus stop coordinates
            bus_stop_coords = bus_stops.geometry.apply(get_coords).dropna()
            bus_stop_nodes = [ox.distance.nearest_nodes(G_roads, X=lon, Y=lat) for lat, lon in bus_stop_coords]

            # Extract 'name' and 'ref' attributes for bus stops (if available)
            bus_stops['name'] = bus_stops.get('name', 'Unknown Name')
            bus_stops['ref'] = bus_stops.get('ref', 'Unknown Ref')

        else:
            
            bus_stops = staion_data

            bus_stops['geometry'] = bus_stops.apply(lambda row: Point(row[station_columns_names['stop_x']], row[station_columns_names['stop_y']]), axis=1)
            bus_stops_geo = gpd.GeoDataFrame(bus_stops, geometry='geometry')

            # assume that the bus station coordinates are in Sweref99 TM
            bus_stops_geo.set_crs(epsg=3006, inplace=True)

            if bus_stops_geo.crs != 'EPSG:4326':
                bus_stops_geo = bus_stops_geo.to_crs('EPSG:4326')
                #print("CRS transformed to EPSG:4326.")
            
            bus_stop_coords = bus_stops_geo.geometry.apply(get_coords).dropna()
            bus_stop_nodes = [ox.distance.nearest_nodes(G_roads, X=lon, Y=lat) for lat, lon in bus_stop_coords]

            bus_stops = bus_stops_geo
        
        # Initialize cost matrix for travel times
        n = len(bus_stop_nodes)
        cost_matrix = np.zeros((n, n))
        results = []
        Count = 1
        
        print('OSM data är nedladdat, börjar beräkningen')

        # Calculate travel times between bus stops
        if not only_download:

            if bus_stop_nodes is None:
                print('No bus stop data, something is strange, check the name of the area in OSM, it need to be exact')
                return

            for i in range(len(bus_stop_nodes)):

                percentage_complete = int(i / len(bus_stop_nodes) * 100)
                sys.stdout.write(f"\r{percentage_complete}%")
                sys.stdout.flush()

                for j in range(len(bus_stop_nodes)):
                    
                    reachable_nodes = nx.single_source_dijkstra_path_length(G_roads, bus_stop_nodes[i], weight='length', cutoff=max_distance)
                    
                    if bus_stop_nodes[j] in reachable_nodes:
                        distance = reachable_nodes[bus_stop_nodes[j]]
                        travel_time = distance / speed # m / m/s = s

                        if staion_data is None:
                            # Get bus stop info
                            bus_stop_i_name = bus_stops.iloc[i]['name']
                            bus_stop_i_ref = bus_stops.iloc[i]['ref']
                            bus_stop_j_name = bus_stops.iloc[j]['name']
                            bus_stop_j_ref = bus_stops.iloc[j]['ref']

                            bus_stop_i_coords = bus_stop_coords.iloc[i]
                            bus_stop_j_coords = bus_stop_coords.iloc[j]

                            if pd.isna(bus_stop_i_ref):
                                bus_stop_1_label = bus_stop_i_name
                            else:
                                bus_stop_1_label = f'{bus_stop_i_name} ({bus_stop_i_ref})'

                            if pd.isna(bus_stop_j_ref):
                                bus_stop_2_label = bus_stop_j_name
                            else:
                                bus_stop_2_label = f'{bus_stop_j_name} ({bus_stop_j_ref})'

                            if bus_stop_1_label != bus_stop_2_label:
                                # Create row with data
                                row = {
                                    'index' : Count,
                                    'bus_stop_1': bus_stop_1_label,
                                    'bus_stop_2': bus_stop_2_label,
                                    'bus_stop_1_lat': bus_stop_i_coords[0],
                                    'bus_stop_1_lon': bus_stop_i_coords[1],
                                    'bus_stop_2_lat': bus_stop_j_coords[0],
                                    'bus_stop_2_lon': bus_stop_j_coords[1],
                                    'distance_m': distance,
                                    'travel_time_sec': travel_time
                                }
                                results.append(row)
                                Count+=1      
                        else:

                            bus_stop_1_label = bus_stops.loc[i, station_columns_names['bus_stop_id']]
                            bus_stop_2_label = bus_stops.loc[j, station_columns_names['bus_stop_id']]

                            bus_stop_1_name = bus_stops.loc[i, station_columns_names['bus_stop_name']]
                            bus_stop_2_name = bus_stops.loc[j, station_columns_names['bus_stop_name']]

                            bus_stop_i_coords = bus_stop_coords.iloc[i]
                            bus_stop_j_coords = bus_stop_coords.iloc[j]

                            if bus_stop_1_label != bus_stop_2_label:
                                # Create row with data
                                row = {
                                    'index' : Count,
                                    'bus_stop_1': bus_stop_1_label,
                                    'bus_stop_2': bus_stop_2_label,
                                    'bus_stop_1_lat': bus_stop_i_coords[0],
                                    'bus_stop_1_lon': bus_stop_i_coords[1],
                                    'bus_stop_2_lat': bus_stop_j_coords[0],
                                    'bus_stop_2_lon': bus_stop_j_coords[1],
                                    'distance_m': distance,
                                    'travel_time_sec': travel_time,
                                    'bus_stop_1_name': bus_stop_1_name,
                                    'bus_stop_2_name': bus_stop_2_name
                                }
                                results.append(row)
                                Count+=1      


            print("Beräkning klar")
            # Convert results to DataFrame
            df_results = pd.DataFrame(results)

            # Save data to CSV
            file_path = os.path.join(output_folder, f"{municipality_name}_bus_stop_travel_times.csv")
            df_results.to_csv(file_path, sep=';', decimal=',', encoding='utf-8',index=False)
            print("CSV fil sparad som 'bus_stop_travel_times.csv'.")
        
        if save_osm_check:
            # Create a dictionary to store all the variables
            data = {
                "G_roads": G_roads,
                "buildings": buildings,
                "waterways": waterways,
                "bus_stops": bus_stops
            }
            # Save the dictionary to a pickle file
            file_path = os.path.join(output_folder, f"{municipality_name}_osm_data.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(data, f)
                print("osm data sparad i pickle format: municipality_osm_data.pkl")

        if plot_check:
            plot_osm_data(G_roads, bus_stops, buildings, waterways, municipality_name)
        
        print('osm_data_run KLAR')
        return True

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

    return 

# Function to plot the map
def plot_osm_data(G_roads, bus_stops, buildings, waterways, municipality_name):
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot buildings as the background layer
    buildings.plot(ax=ax, color="lightgray", edgecolor="black", linewidth=0.5, alpha=0.7)

    # Plot the main road network
    ox.plot_graph(G_roads, ax=ax, show=False, close=False, edge_color="blue", edge_linewidth=0.7, node_size=0)

    # Plot waterways
    waterways.plot(ax=ax, color="aqua", linewidth=1.0, label="Waterways")

    # Plot bus stops
    bus_stops.plot(ax=ax, color="red", markersize=10, label="Bus Stops")

    # Customize and show the plot
    ax.set_title(f"Road Network, Buildings, Waterways, and Bus Stops in {municipality_name}")
    plt.show()

