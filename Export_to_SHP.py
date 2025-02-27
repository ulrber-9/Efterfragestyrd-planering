import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import osmnx as ox

import folium
from folium.plugins import FloatImage
import branca

# Data Paths
main_path = r"C:/Users/CJ/OneDrive - Lund University/Lund/Ulrik projekt/Data/Västtrafik/Originalfiler/"

# Input File Names
Output_OD_Matrix_file = "Output_OD_Matrix.csv"
stopkey_file = "StopKey.csv"
pickle_osm_data_file = "Alingsås kommun_osm_data.pkl"

# Read data
OD_matrix = pd.read_csv(os.path.join(main_path, Output_OD_Matrix_file),sep=';') # ;ValidationDate;BoardingStop;Final_AlightingStop;count
Stopkeydata = pd.read_csv(os.path.join(main_path, stopkey_file))

# Station location columns. This maps column names in the file to the names used in the code
station_location_cols = {
    'bus_stop_name': 'HplNamn',
    'bus_stop_id': 'HplNUmmer',
    'stop_x': 'xcoord',
    'stop_y': 'ycoord',
}

# Rename columns in Stopkeydata using the mapping dictionary
column_mapping = {v: k for k, v in station_location_cols.items()}
Stopkeydata = Stopkeydata.rename(columns=column_mapping)

# Load OSM data
with open(os.path.join(main_path, pickle_osm_data_file), "rb") as file:
    OSM_data = pickle.load(file)

# Merge OD data with stop coordinates for boarding stops
OD_with_coords = OD_matrix.merge(
    Stopkeydata[['bus_stop_id', 'bus_stop_name', 'stop_x', 'stop_y']],
    left_on='BoardingStop',
    right_on='bus_stop_id',
    how='left',
    suffixes=('', '_boarding')
)

# Rename columns to avoid confusion after the first merge
OD_with_coords = OD_with_coords.rename(columns={
    'bus_stop_name': 'boarding_stop_name',
    'stop_x': 'boarding_stop_x',
    'stop_y': 'boarding_stop_y'
})

# Merge again for alighting stops
OD_with_coords = OD_with_coords.merge(
    Stopkeydata[['bus_stop_id', 'bus_stop_name', 'stop_x', 'stop_y']],
    left_on='Final_AlightingStop',
    right_on='bus_stop_id',
    how='left',
    suffixes=('', '_alighting')
)

# Rename columns from the second merge
OD_with_coords = OD_with_coords.rename(columns={
    'bus_stop_name': 'alighting_stop_name',
    'stop_x': 'alighting_stop_x',
    'stop_y': 'alighting_stop_y',
    'count': 'trips_day'
})

# Group by stop pairs and sum counts across all ValidationDate values
OD_with_coords["total_trips_for_all_days"] = OD_with_coords.groupby(["BoardingStop", "Final_AlightingStop"])["trips_day"].transform("sum")

# Convert coordinates to float if they are strings
for col in ['boarding_stop_x', 'boarding_stop_y', 'alighting_stop_x', 'alighting_stop_y']:
    if OD_with_coords[col].dtype == 'object':
        OD_with_coords[col] = OD_with_coords[col].str.replace(',', '.').astype(float)

# Before creating LineString geometries, filter out rows with missing coordinates
OD_with_coords = OD_with_coords.dropna(subset=['boarding_stop_x', 'boarding_stop_y', 'alighting_stop_x', 'alighting_stop_y'])

# Create LineString geometries for OD pairs in SWEREF99TM (EPSG:3006)
OD_with_coords['geometry'] = OD_with_coords.apply(
    lambda row: LineString([
        (row['boarding_stop_x'], row['boarding_stop_y']),
        (row['alighting_stop_x'], row['alighting_stop_y'])
    ]),
    axis=1
)

# Remove rows where boarding stop == alighting stop
OD_with_coords = OD_with_coords[
    OD_with_coords['BoardingStop'] != OD_with_coords['Final_AlightingStop']
]

# Create GeoDataFrame with SWEREF99TM projection
OD_gdf = gpd.GeoDataFrame(OD_with_coords, geometry='geometry', crs="EPSG:3006")

# Create bus stops GeoDataFrame
bus_stops = Stopkeydata.copy()

# Ensure bus_stop_id is a string for consistency
bus_stops["bus_stop_id"] = bus_stops["bus_stop_id"].astype(str)

# Create a set of valid stop IDs from both columns
valid_bus_stops = set(OD_with_coords["BoardingStop"].dropna().astype(int).astype(str)) | \
                  set(OD_with_coords["Final_AlightingStop"].dropna().astype(int).astype(str))

# Filter bus_stops to keep only those appearing in OD_with_coords
bus_stops = bus_stops[bus_stops["bus_stop_id"].isin(valid_bus_stops)]

bus_stops['geometry'] = bus_stops.apply(lambda row: Point(row['stop_x'], row['stop_y']), axis=1)
bus_stops_gdf = gpd.GeoDataFrame(bus_stops, geometry='geometry', crs="EPSG:3006")

# Extract OSM layers
roads_gdf = OSM_data.get("G_roads", None)
osm_bus_stops_gdf = OSM_data.get("bus_stops", None)
buildings_gdf = OSM_data.get("buildings", None)

# Define outputs directory
output_dir = os.path.join(main_path, "outputs")
os.makedirs(output_dir, exist_ok=True)

# Export to Shapefile (already in SWEREF99TM)
OD_gdf.to_file(os.path.join(output_dir, "od_flows.shp"))
bus_stops_gdf.to_file(os.path.join(output_dir, "bus_stops.shp"))

# Export OSM layers if they exist
if roads_gdf is not None:
    roads_gdf = ox.graph_to_gdfs(roads_gdf, nodes=False, edges=True)

    if roads_gdf.crs != "EPSG:3006":
        roads_gdf = roads_gdf.to_crs("EPSG:3006")
        
    roads_gdf.to_file(os.path.join(output_dir, "roads.shp"))
    
if buildings_gdf is not None:
    if buildings_gdf.crs != "EPSG:3006":
        buildings_gdf = buildings_gdf.to_crs("EPSG:3006")

    buildings_gdf = buildings_gdf[buildings_gdf.geom_type.isin(["Polygon", "MultiPolygon"])]
    buildings_gdf.to_file(os.path.join(output_dir, "buildings.shp"),driver="ESRI Shapefile")
    
print("Done with exports!")

# Folium dynamic map

OD_gdf.drop_duplicates(subset=['BoardingStop','Final_AlightingStop'])

roads_gdf = roads_gdf.to_crs(epsg=4326)
buildings_gdf = buildings_gdf.to_crs(epsg=4326)
bus_stops_gdf = bus_stops_gdf.to_crs(epsg=4326)
OD_gdf = OD_gdf.to_crs(epsg=4326)

# Normalize width and color (default range)
def scale_width(value, min_width=1, max_width=30):
    return np.interp(value, (OD_gdf["total_trips_for_all_days"].min(), OD_gdf["total_trips_for_all_days"].max()), (min_width, max_width))

OD_gdf["width"] = OD_gdf["total_trips_for_all_days"].apply(scale_width)

# Normalize color (colormap from viridis)
def get_color(value):
    norm_val = value / OD_gdf["total_trips_for_all_days"].max()  # Normalize
    rgb = plt.cm.viridis(norm_val, bytes=True)[:3]  # Get RGB values
    return "#{:02x}{:02x}{:02x}".format(*rgb)  # Convert to hex

OD_gdf["color"] = OD_gdf["total_trips_for_all_days"].apply(get_color)

# Initialize map
m = folium.Map(location=[bus_stops_gdf.geometry.y.mean(), bus_stops_gdf.geometry.x.mean()], zoom_start=13, tiles="cartodb positron")

# Add roads as a background layer
for _, row in roads_gdf.iterrows():

    coords = list(row.geometry.coords)  # This is a list of (longitude, latitude) tuples

    folium.PolyLine(
        locations=[(lat, lon) for lon, lat in coords],  # Adjusting (lat, lon) order
        color="black",
        weight=1,
        opacity=0.5
    ).add_to(m)

# Add buildings as a background layer
for _, row in buildings_gdf.iterrows():
    folium.GeoJson(row.geometry, style_function=lambda x: {"fillColor": "grey", "color": "grey", "weight": 0.5, "fillOpacity": 0.3}).add_to(m)

# Add bus stops
for _, row in bus_stops_gdf.iterrows():
    folium.CircleMarker(
        location=(row.geometry.y, row.geometry.x),
        radius=5,
        color="red",
        fill=True,
        fill_color="red",
        fill_opacity=1,
        popup=f"Bus Stop ID: {row['bus_stop_id']}"  # Adjust based on your column names
    ).add_to(m)

# Add OD matrix flows
for _, row in OD_gdf.iterrows():

    coords = list(row.geometry.coords)  # This is a list of (longitude, latitude) tuples

    line = folium.PolyLine(
        locations=[(lat, lon) for lon, lat in coords],  # Adjusting (lat, lon) order
        color=row["color"],
        weight=row["width"],
        opacity=0.7
    ).add_to(m)


# Save the map
m.save(os.path.join(output_dir, "od_matrix_map.html" ))
print("All done!")