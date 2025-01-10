import OSM_BUS_STOP_PATHS as bs
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz  # NOTE: If not used anywhere else, you can remove this import
import geopandas as gpd
from shapely.geometry import LineString
import os
import pickle
import matplotlib.pyplot as plt
import osmnx as ox

############################################
# Configuration and Column Name Mappings
############################################

# Basic Parameters
municipality_name = 'Alingsås kommun'
walking_speed = 5       # km/h - used to calculate potential walking travel times or limits
limit_distance = 500    # m - maximum allowed walking distance
Plot_Check = True       # Whether to plot final results or not

only_select_lines = True
only_these_lines = [6570, 6571, 6572, 6573, 6574]

# Data Paths
mainPath = r"C:/Users/CJ/OneDrive - Lund University/Lund/Ulrik projekt/Data/Västtrafik/Originalfiler/"

# Input File Names
ticket_validations_file = "TicketValidations.csv"
realtidsdata_file = "Realtidsdata.csv"
stopkey_file = "StopKey.csv"

# Column Mappings for Input Data
ticket_cols = {
    'traveller_id': 'TravellerId',
    'validation_date': 'ValidationDate',
    'validation_time': 'ValidationTime',
    'stop_area_number': 'StopAreaNumber',
    'technical_line_number': 'TechnicalLineNumber'
}

realtime_cols = {
    'stop_area_name': 'StopAreaName',
    'stop_area_number': 'StopAreaNumber',
    'trip_no': 'TripNo',
    'operating_day_date': 'OperatingDayDate',
    'planned_departure_time': 'PlannedDepartureTime',
    'actual_arrival_time': 'ActualArrivalTime',
    'actual_departure_time': 'ActualDepartureTime',
    'technical_line_number': 'TechnicalLineNumber'
}

# Station location columns (used for OSM data extraction)
station_location_cols = {
    'bus_stop_name': 'HplNamn',
    'bus_stop_id': 'HplNUmmer',
    'stop_x': 'xcoord',
    'stop_y': 'ycoord',
}

# Output File Names
output_od_matrix_file = "Output_OD_Matrix.csv"


############################################
# Helper Functions
############################################

def run_osm_data_if_needed(municipality_name, output_folder, speed, max_distance, Stopkeydata, station_location_cols):
    """
    Check if required OSM (OpenStreetMap) data files already exist.
    If not, run the OSM data extraction routine from `OSM_BUS_STOP_PATHS.osm_data_run`.

    The function looks for:
       - <municipality_name>_bus_stop_travel_times.csv
       - <municipality_name>_osm_data.pkl
    If these are not found, it runs `bs.osm_data_run` to generate them.

    Parameters:
    -----------
    municipality_name : str
        Name of the municipality (e.g., 'Alingsås kommun').
    output_folder : str
        Path to the folder where data should be stored and read from.
    speed : float
        Walking speed in km/h.
    max_distance : float
        Maximum allowed walking distance in meters.
    Stopkeydata : pd.DataFrame
        DataFrame with stop key information (bus stop IDs, names, etc.).
    station_location_cols : dict
        Column name mapping for station location data.
    """
    # Paths for expected output files
    stop_matrix_file = os.path.join(output_folder, f"{municipality_name}_bus_stop_travel_times.csv")
    osm_pickle_file = os.path.join(output_folder, f"{municipality_name}_osm_data.pkl")

    # Check existence of files
    if os.path.exists(stop_matrix_file) and os.path.exists(osm_pickle_file):
        print("\nOSM data files already exist. Skipping `bs.osm_data_run`.\n")
    else:
        print("\nOSM data files not found. Running `bs.osm_data_run`.\n")
        # Run data generation
        bs.osm_data_run(
            municipality_name,
            output_folder=output_folder,
            speed=speed,
            max_distance=max_distance,
            plot_check=False,
            staion_data=Stopkeydata,
            station_columns_names=station_location_cols
        )
        print("\nOSM data generation completed.\n")


def load_data(main_path, ticket_file, realtime_file, stopkey_file):
    """
    Load Ticket Validations, Realtime data, and StopKey data from CSV files into DataFrames.
    Optionally filter lines if `only_select_lines` is set to True.

    Parameters:
    -----------
    main_path : str
        Main folder path containing the CSV files.
    ticket_file : str
        Name of the ticket validations file.
    realtime_file : str
        Name of the realtime data file.
    stopkey_file : str
        Name of the stopkey file.

    Returns:
    --------
    TicketValidations : pd.DataFrame
        Ticket validations dataset.
    Realtidsdata : pd.DataFrame
        Realtime data dataset.
    Stopkeydata : pd.DataFrame
        Stop key dataset.
    """
    TicketValidations = pd.read_csv(os.path.join(main_path, ticket_file))
    Realtidsdata = pd.read_csv(os.path.join(main_path, realtime_file))
    Stopkeydata = pd.read_csv(os.path.join(main_path, stopkey_file))

    # If desired, filter only the specified lines
    if only_select_lines:
        TicketValidations = TicketValidations[
            TicketValidations[ticket_cols['technical_line_number']].isin(only_these_lines)
        ]
        Realtidsdata = Realtidsdata[
            Realtidsdata[realtime_cols['technical_line_number']].isin(only_these_lines)
        ]

    return TicketValidations, Realtidsdata, Stopkeydata


def prepare_realtidsdata(Realtidsdata, cols):
    """
    Clean and prepare the Realtime data (Realtidsdata):
      - Convert times to datetime
      - Replace missing ActualArrivalTime with ActualDepartureTime where needed
      - Create a 'StopSequence' column to indicate the order of stops for each trip

    Parameters:
    -----------
    Realtidsdata : pd.DataFrame
        The realtime data DataFrame.
    cols : dict
        Dictionary mapping of column names for realtime data.

    Returns:
    --------
    Realtidsdata : pd.DataFrame
        Cleaned and updated realtime DataFrame.
    """
    # Convert planned departure time to datetime
    Realtidsdata[cols['planned_departure_time']] = pd.to_datetime(
        Realtidsdata[cols['planned_departure_time']], format='%H:%M'
    )

    # Fill ActualArrivalTime with ActualDepartureTime if ActualArrivalTime is missing
    Realtidsdata[cols['actual_arrival_time']] = Realtidsdata[cols['actual_arrival_time']].fillna(
        Realtidsdata[cols['actual_departure_time']]
    )

    # Extract departure hour from planned departure time
    Realtidsdata['dep_hour'] = Realtidsdata[cols['planned_departure_time']].dt.hour

    # Create a StopSequence to show the order of stops within a trip and day
    Realtidsdata['StopSequence'] = (
        (Realtidsdata[cols['trip_no']] != Realtidsdata[cols['trip_no']].shift())
        | (Realtidsdata[cols['operating_day_date']] != Realtidsdata[cols['operating_day_date']].shift())
    ).cumsum()
    Realtidsdata['StopSequence'] = Realtidsdata.groupby(
        [cols['trip_no'], cols['operating_day_date']]
    ).cumcount() + 1

    return Realtidsdata


def prepare_ticket_validations(TicketValidations, cols):
    """
    Prepare Ticket Validations data:
      - Remove invalid traveller IDs (like -1)
      - Sort by traveller and time
      - Rename TechnicalLineNumber column to 'Linjenummer'
      - Add a unique observation counter (obs)

    Parameters:
    -----------
    TicketValidations : pd.DataFrame
        The ticket validations DataFrame.
    cols : dict
        Dictionary mapping for ticket validation column names.

    Returns:
    --------
    TicketValidations : pd.DataFrame
        Cleaned and updated ticket validations DataFrame.
    """
    # Filter out invalid traveller IDs
    TicketValidations = TicketValidations[
        TicketValidations[cols['traveller_id']] != -1
    ]

    # Sort by traveller, date, time
    TicketValidations = TicketValidations.sort_values(
        by=[cols['traveller_id'], cols['validation_date'], cols['validation_time']]
    )

    # Rename TechnicalLineNumber -> Linjenummer
    TicketValidations.rename(
        columns={cols['technical_line_number']: 'Linjenummer'},
        inplace=True
    )

    # Add an observation index for each row
    TicketValidations['obs'] = np.arange(1, len(TicketValidations) + 1)

    return TicketValidations


def create_stop_combinations(Realtidsdata, cols):
    """
    From the Realtime data, create a time-expanded stop matrix:
    For each trip, pair every stop with every other stop (where sequence_from <= sequence_to).
    This will give possible boarding-alighting combinations per trip and day.

    Parameters:
    -----------
    Realtidsdata : pd.DataFrame
        The realtime data DataFrame.
    cols : dict
        Dictionary mapping for realtime column names.

    Returns:
    --------
    stopcomb_times : pd.DataFrame
        DataFrame with columns for from-stop, to-stop, arrival, departure, etc.
    """
    # Extract required columns to identify line/trip/day/stop/time
    linestops = Realtidsdata[
        [
            cols['technical_line_number'],
            cols['trip_no'],
            cols['operating_day_date'],
            cols['stop_area_number'],
            'StopSequence',
            cols['actual_departure_time'],
            cols['actual_arrival_time']
        ]
    ].drop_duplicates()

    # Self-join to get pairs of stops within the same line/trip/day
    stopcomb_times = linestops.merge(
        linestops,
        on=[
            cols['technical_line_number'],
            cols['trip_no'],
            cols['operating_day_date']
        ],
        suffixes=('_from', '_to')
    )

    # Keep pairs where 'from' stop comes before or is the same as 'to' stop
    stopcomb_times = stopcomb_times[
        stopcomb_times['StopSequence_from'] <= stopcomb_times['StopSequence_to']
    ]

    # Keep and rename relevant columns
    stopcomb_times = stopcomb_times[
        [
            cols['technical_line_number'],
            cols['trip_no'],
            cols['operating_day_date'],
            f"{cols['stop_area_number']}_from",
            f"{cols['stop_area_number']}_to",
            f"{cols['actual_departure_time']}_from",
            f"{cols['actual_arrival_time']}_to"
        ]
    ]

    stopcomb_times.rename(
        columns={
            f"{cols['stop_area_number']}_from": 'FromStp',
            f"{cols['stop_area_number']}_to": 'ToStp',
            f"{cols['actual_departure_time']}_from": 'ActualDepartureTime',
            f"{cols['actual_arrival_time']}_to": 'ActualArrivalTime'
        },
        inplace=True
    )

    # Convert OperatingDayDate to datetime
    stopcomb_times['OperatingDayDate'] = pd.to_datetime(
        stopcomb_times['OperatingDayDate'].astype(str), format='%Y%m%d'
    )

    # Filter out rows with missing stops
    stopcomb_times = stopcomb_times[
        (stopcomb_times['FromStp'].notna()) & (stopcomb_times['ToStp'].notna())
    ]

    return stopcomb_times


def get_target_stops(TicketValidations, StopMatrix):
    """
    For each ticket validation record, find the next possible target stop.
    Uses a lag (shift) operation on TicketValidations to align with next records.
    Then merges with the StopMatrix to match bus stop numbers.

    Parameters:
    -----------
    TicketValidations : pd.DataFrame
        Cleaned ticket validation records.
    StopMatrix : pd.DataFrame
        Precomputed matrix with travel times/distances between bus stops.

    Returns:
    --------
    TicketValidations_target : pd.DataFrame
        Ticket validations with an additional column 'stp_t' (potential next stop),
        merged to the StopMatrix for matching bus stops.
    """
    # Sort descending by obs, shift, then resort ascending to align the next stop as 'stp_t'
    TicketValidations_target = TicketValidations.sort_values(by='obs', ascending=False).copy()
    TicketValidations_target['stp_t'] = TicketValidations_target['StopAreaNumber'].shift(1)
    TicketValidations_target = TicketValidations_target.sort_values(by='obs')

    # Merge to find matches in StopMatrix for the 'stp_t' stop (potential next stops)
    TicketValidations_target = TicketValidations_target.merge(
        StopMatrix,
        left_on='stp_t',
        right_on='matched_bus_stop_1_number',
        how='inner'
    )

    return TicketValidations_target


def generate_possible_targets(TicketValidations_target):
    """
    Prepare a DataFrame of possible target stops for each ticket validation.
    Sort records and ensure proper datetime format for ValidationDate.

    Parameters:
    -----------
    TicketValidations_target : pd.DataFrame
        DataFrame returned from get_target_stops().

    Returns:
    --------
    PossibleTargetStops : pd.DataFrame
        Sorted and cleaned DataFrame with possible target stops.
    """
    PossibleTargetStops = TicketValidations_target.copy()
    # Sort by date, line number, original stop number, and matched target stop
    PossibleTargetStops = PossibleTargetStops.sort_values(
        by=['ValidationDate', 'Linjenummer', 'StopAreaNumber', 'matched_bus_stop_2_number']
    )

    # Ensure ValidationDate is in datetime format
    PossibleTargetStops['ValidationDate'] = pd.to_datetime(
        PossibleTargetStops['ValidationDate'].astype(str), format='%Y-%m-%d'
    )

    return PossibleTargetStops


def match_targets_to_trips(PossibleTargetStops, stopcomb_times):
    """
    Match the candidate target stops identified from the StopMatrix with actual trip data from stopcomb_times.
    This identifies which actual trip segments could correspond to a passenger's next (alighting) stop.

    Parameters:
    -----------
    PossibleTargetStops : pd.DataFrame
        DataFrame of potential next stops for each ticket validation record.
    stopcomb_times : pd.DataFrame
        Time-expanded stop combination data (boarding/alighting pairs).

    Returns:
    --------
    AlightingStop : pd.DataFrame
        DataFrame with the best-matching alighting stop (and trip segment) for each record.
    """
    AlightingStop_temp = pd.merge(
        PossibleTargetStops,
        stopcomb_times,
        how='outer',
        left_on=['StopAreaNumber', 'matched_bus_stop_2_number', 'ValidationDate', 'Linjenummer'],
        right_on=['FromStp', 'ToStp', 'OperatingDayDate', 'TechnicalLineNumber'],
        indicator=True
    )

    # Keep only matched rows
    AlightingStop_temp = AlightingStop_temp[AlightingStop_temp['_merge'] == 'both'].copy()
    AlightingStop_temp.drop(columns=['_merge'], inplace=True)

    # Rename columns for clarity
    AlightingStop_temp['BoardingStop'] = AlightingStop_temp['FromStp']
    AlightingStop_temp['AlightingStop'] = AlightingStop_temp['ToStp']

    # Convert ValidationTime and ActualDepartureTime to datetime format
    AlightingStop_temp['ValidationTime'] = pd.to_datetime(
        AlightingStop_temp['ValidationTime'], format='%H:%M'
    )
    AlightingStop_temp['ActualDepartureTime'] = pd.to_datetime(
        AlightingStop_temp['ActualDepartureTime'], format='%H:%M'
    )

    # Calculate absolute time difference between validation time and actual departure time
    AlightingStop_temp['timediff'] = (
        AlightingStop_temp['ValidationTime'] - AlightingStop_temp['ActualDepartureTime']
    ).abs()

    # Sort by timediff (and other keys) to pick the best match
    AlightingStop = AlightingStop_temp.sort_values(
        by=['TravellerId', 'ValidationDate', 'ValidationTime', 'obs',
            'TechnicalLineNumber', 'BoardingStop', 'AlightingStop', 'timediff']
    )

    # Keep only the first record per group (minimal timediff)
    AlightingStop = AlightingStop.groupby(
        ['TravellerId', 'ValidationDate', 'ValidationTime', 'obs',
         'TechnicalLineNumber', 'BoardingStop', 'AlightingStop'],
        as_index=False
    ).first()

    # Remove cases where boarding stop and alighting stop are the same
    AlightingStop = AlightingStop[AlightingStop['BoardingStop'] != AlightingStop['AlightingStop']]

    # If multiple results per boarding stop remain, keep the first
    AlightingStop = AlightingStop.sort_values(
        by=['TravellerId', 'ValidationDate', 'ValidationTime', 'obs',
            'TechnicalLineNumber', 'BoardingStop']
    )
    AlightingStop = AlightingStop.groupby(
        ['TravellerId', 'ValidationDate', 'ValidationTime', 'obs',
         'TechnicalLineNumber', 'BoardingStop'],
        as_index=False
    ).first()

    # Compute actual transfer times
    AlightingStop['VAnkT_prev_run'] = AlightingStop['ActualArrivalTime'].shift(1)
    AlightingStop['VAnkT_prev_run'] = pd.to_datetime(
        AlightingStop['VAnkT_prev_run'], format='%H:%M'
    )
    AlightingStop['T_Act_transfer'] = AlightingStop['ActualDepartureTime'] - AlightingStop['VAnkT_prev_run']

    # Determine day type (weekday vs. weekend)
    AlightingStop['DayOfWeek'] = AlightingStop['OperatingDayDate'].dt.weekday + 1  # Monday=1, Sunday=7
    AlightingStop['daytype'] = AlightingStop['DayOfWeek'].apply(lambda x: 2 if x in [6, 7] else 1)

    # Sort by daytype, date, traveller
    AlightingStop = AlightingStop.sort_values(by=['daytype', 'ValidationDate', 'TravellerId'])

    return AlightingStop


def find_headway(row, Realtidsdata):
    """
    Helper function to calculate the headway for a given row (based on line, date, and end stop).
    Finds up to three closest buses (in time) for that stop, line, date and computes the average interval.
    """
    val_date = row['ValidationDate']
    val_line = row['TechnicalLineNumber']
    val_end_stop = row['EndStopAreaNumber']
    val_dt = row['ValidationDatetime']

    # Filter to same date, line, and end stop
    mask = (
        (Realtidsdata['ActualDepartureDate'] == val_date)
        & (Realtidsdata['TechnicalLineNumber'] == val_line)
        & (Realtidsdata['StopAreaNumber'] == val_end_stop)
    )
    candidates = Realtidsdata.loc[mask].copy()

    # Calculate absolute time difference from ValidationDatetime
    candidates['time_diff'] = (candidates['ActualArrivalDatetime'] - val_dt).abs()

    # Sort by the absolute time difference and select the three closest
    closest_buses = candidates.sort_values('time_diff').head(3)

    # If fewer than 3 buses found, can't compute a stable headway
    if len(closest_buses) < 3:
        return np.nan

    # Compute intervals between the three times
    arrival_times = closest_buses['ActualArrivalDatetime'].sort_values().reset_index(drop=True)
    intervals = arrival_times.diff().dropna().dt.total_seconds()

    # If we don't have two intervals (which we should for three times), return NaN
    if len(intervals) < 2:
        return np.nan

    # Headway is the mean of these intervals in seconds
    headway = intervals.mean()
    return headway


def calculate_headways(AlightingStop, Realtidsdata, cols):
    """
    Calculate headways (average time intervals between trips on the same stop, line, and day) 
    and merge these headways onto the AlightingStop data.

    Parameters:
    -----------
    AlightingStop : pd.DataFrame
        DataFrame with matched alighting stops for each passenger record.
    Realtidsdata : pd.DataFrame
        The realtime data DataFrame with actual departure/arrival times and dates.
    cols : dict
        Dictionary mapping for realtime column names.

    Returns:
    --------
    AlightingStop : pd.DataFrame
        Updated DataFrame with an additional 'Headway' column.
    """
    # Create a datetime column in AlightingStop for the validation moment
    AlightingStop['ValidationDatetime'] = pd.to_datetime(
        AlightingStop['ValidationDate'].astype(str) + ' ' + AlightingStop['ValidationTime'].astype(str),
        utc=True,
        errors='coerce'
    )

    # Create a datetime column in Realtidsdata for actual departure
    Realtidsdata['ActualArrivalDatetime'] = pd.to_datetime(
        Realtidsdata['ActualDepartureDate'].astype(str) + ' ' + Realtidsdata['ActualDepartureTime'].astype(str),
        utc=True,
        errors='coerce'
    )

    # Convert 'ActualDepartureDate' to a proper date
    Realtidsdata['ActualDepartureDate'] = pd.to_datetime(
        Realtidsdata['ActualDepartureDate'].astype(str), format='%Y%m%d'
    )

    # Apply find_headway() row by row
    AlightingStop['Headway'] = AlightingStop.apply(
        lambda row: find_headway(row, Realtidsdata),
        axis=1
    )

    return AlightingStop


def determine_final_alighting_stops(AlightingStop):
    """
    Determine the final alighting stops in a travel chain considering possible transfers.
    If a passenger transfers within a short time (<= headway), the final alighting might be 
    a further stop on the next trip.

    Parameters:
    -----------
    AlightingStop : pd.DataFrame
        DataFrame with matched alighting stops.

    Returns:
    --------
    Itinerary : pd.DataFrame
        Updated DataFrame indicating final alighting stops (accounting for transfers).
    """
    Itinerary = AlightingStop.copy()
    Itinerary = Itinerary.sort_values(by='obs')

    # Identify previous date and next date records for checking same-day transfers
    Itinerary['prevdate'] = Itinerary['ValidationDate'].shift(1)
    Itinerary = Itinerary.sort_values(by='obs', ascending=False)
    Itinerary['NextDate'] = Itinerary['ValidationDate'].shift(1)
    Itinerary = Itinerary.sort_values(by='obs')

    # Act_Transfer1: Check if the next boarding (previous in obs order) is on the same date
    Itinerary['Act_Transfer1'] = np.where(Itinerary['prevdate'] == Itinerary['ValidationDate'], 2, 1)

    # Act_Transfer2: Check if actual transfer time is less or equal to headway
    Itinerary['Act_Transfer2'] = np.where(
        ((Itinerary['T_Act_transfer'].dt.seconds / 60) <= Itinerary['Headway']),
        2, 
        1
    )

    # Combine conditions: Act_Transfer=2 means a transfer actually happened
    Itinerary['Act_Transfer'] = np.where(
        (Itinerary['Act_Transfer1'] == 2) & (Itinerary['Act_Transfer2'] == 2),
        2, 
        1
    )

    # Sort by date and traveller to align next steps
    Itinerary = Itinerary.sort_values(by=['ValidationDate', 'TravellerId'])

    # Sort by obs descending to look ahead to the next record
    Itinerary = Itinerary.sort_values(by='obs', ascending=False)
    Itinerary['Next_ActTransfer'] = Itinerary['Act_Transfer'].shift(1)
    Itinerary['Next_AlightingStop'] = Itinerary['AlightingStop'].shift(1)
    Itinerary['next_Verklig_ankomsttid'] = Itinerary['ActualArrivalTime'].shift(1)
    Itinerary = Itinerary.sort_values(by='obs')

    # If next step is a transfer and same date, final alighting is that next step's alighting stop
    Itinerary['Final_AlightingStop'] = np.where(
        (Itinerary['Next_ActTransfer'] == 2) & (Itinerary['ValidationDate'] == Itinerary['NextDate']),
        Itinerary['Next_AlightingStop'],
        Itinerary['AlightingStop']
    )
    Itinerary['Final_Verklig_ankomsttid'] = np.where(
        (Itinerary['Next_ActTransfer'] == 2) & (Itinerary['ValidationDate'] == Itinerary['NextDate']),
        Itinerary['next_Verklig_ankomsttid'],
        Itinerary['ActualArrivalTime']
    )

    return Itinerary


def create_od_matrix(Itinerary):
    """
    Create an Origin-Destination (OD) matrix by day:
    Count how many travellers boarded at a stop and finally alighted at another stop on a given date.

    Parameters:
    -----------
    Itinerary : pd.DataFrame
        DataFrame with final alighting stops.

    Returns:
    --------
    OD_matrix_By_Day : pd.DataFrame
        Grouped DataFrame with columns:
            - ValidationDate
            - BoardingStop
            - Final_AlightingStop
            - count (number of passengers)
    """
    # Select records representing final trips (no further transfers)
    proto_ODmatrix = Itinerary[
        (Itinerary['Act_Transfer'] == 1) &
        (Itinerary['BoardingStop'] != Itinerary['Final_AlightingStop'])
    ]

    proto_ODmatrix = proto_ODmatrix.sort_values(
        by=['ValidationDate', 'BoardingStop', 'Final_AlightingStop']
    )

    OD_matrix_By_Day = (
        proto_ODmatrix
        .groupby(['ValidationDate', 'BoardingStop', 'Final_AlightingStop'])
        .size()
        .reset_index(name='count')
    )
    return OD_matrix_By_Day


def plot_osm_data(G_roads, bus_stops, buildings, waterways, municipality_name, OD_with_from_gdf=None):
    """
    Plot OSM data for the municipality:
      - Buildings in lightgray
      - Road network (G_roads) in blue
      - Waterways in aqua
      - Bus stops in red
    Optionally, if OD_with_from_gdf is given, plot OD lines as well (thicker lines for higher counts).

    Parameters:
    -----------
    G_roads : networkx.MultiDiGraph
        Road network graph.
    bus_stops : gpd.GeoDataFrame
        GeoDataFrame of bus stops.
    buildings : gpd.GeoDataFrame
        GeoDataFrame of buildings.
    waterways : gpd.GeoDataFrame
        GeoDataFrame of waterways.
    municipality_name : str
        Name of the municipality.
    OD_with_from_gdf : gpd.GeoDataFrame (optional)
        GeoDataFrame of OD lines with a 'linewidth' column for plotting thickness.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot buildings
    buildings.plot(ax=ax, color="lightgray", edgecolor="black", linewidth=0.5, alpha=0.7)
    # Plot roads
    ox.plot_graph(
        G_roads, ax=ax, show=False, close=False,
        edge_color="blue", edge_linewidth=0.7, node_size=0
    )
    # Plot waterways
    waterways.plot(ax=ax, color="aqua", linewidth=1.0, label="Waterways")
    # Plot bus stops
    bus_stops.plot(ax=ax, color="red", markersize=10, label="Bus Stops")

    # Plot OD lines if provided
    if OD_with_from_gdf is not None:
        OD_with_from_gdf.plot(
            ax=ax, color="green", linewidth=OD_with_from_gdf["linewidth"], alpha=0.7
        )

    ax.set_title(f"Road Network, Buildings, Waterways, and Bus Stops in {municipality_name}")
    plt.show()


############################################
# Main Workflow
############################################

def main(plot_bool=True):
    """
    Main function orchestrating the entire workflow:
      1. Load ticket validations and realtime data.
      2. Run OSM data extraction if needed.
      3. Load the precomputed StopMatrix.
      4. Prepare/clean the realtime data.
      5. Prepare/clean the ticket validations data.
      6. Create time-expanded stop combinations.
      7. For each ticket validation, find next possible target stops.
      8. Match these targets to actual trips to identify actual alighting stops.
      9. Calculate headways to assist in understanding transfers.
      10. Determine final alighting stops considering potential transfers.
      11. Create an OD matrix by day from the final itineraries.
      12. (Optional) Plot results if `plot_bool` is True.
    """
    print('\nStarting program\n')

    # 1. Load ticket validations and realtime data
    TicketValidations, Realtidsdata, Stopkeydata = load_data(
        mainPath,
        ticket_validations_file,
        realtidsdata_file,
        stopkey_file
    )

    # 2. Run OSM data extraction if needed
    run_osm_data_if_needed(
        municipality_name,
        mainPath,
        walking_speed,
        limit_distance,
        Stopkeydata,
        station_location_cols
    )

    # 3. Load precomputed StopMatrix (travel times/distances between stops)
    stop_matrix_path = os.path.join(
        mainPath,
        f"{municipality_name}_bus_stop_travel_times.csv"
    )
    StopMatrix = pd.read_csv(stop_matrix_path, sep=';')

    # Rename columns to align with usage
    StopMatrix.rename(
        columns={'bus_stop_1': 'matched_bus_stop_1_number'},
        inplace=True
    )
    StopMatrix.rename(
        columns={'bus_stop_2': 'matched_bus_stop_2_number'},
        inplace=True
    )

    # 4. Prepare and clean Realtidsdata
    Realtidsdata = prepare_realtidsdata(Realtidsdata, realtime_cols)

    # 5. Prepare and clean TicketValidations
    TicketValidations = prepare_ticket_validations(TicketValidations, ticket_cols)

    # 6. Create time-expanded stop combinations (trip segments)
    stopcomb_times = create_stop_combinations(Realtidsdata, realtime_cols)

    # 7. For each ticket validation, find possible target stops
    TicketValidations_target = get_target_stops(TicketValidations, StopMatrix)

    # 8. Generate a DataFrame of possible target stops
    PossibleTargetStops = generate_possible_targets(TicketValidations_target)

    # 9. Match these targets with actual trips to identify actual alighting stops
    AlightingStop = match_targets_to_trips(PossibleTargetStops, stopcomb_times)

    # 10. Calculate headways (service intervals)
    AlightingStop = calculate_headways(AlightingStop, Realtidsdata, realtime_cols)

    # 11. Determine final alighting stops considering transfers
    Itinerary = determine_final_alighting_stops(AlightingStop)

    # 12. Create OD matrix by day from final itineraries
    OD_matrix_By_Day = create_od_matrix(Itinerary)
    OD_matrix_By_Day = OD_matrix_By_Day[
        OD_matrix_By_Day['BoardingStop'] != OD_matrix_By_Day['Final_AlightingStop']
    ]
    OD_matrix_By_Day.to_csv(os.path.join(mainPath, output_od_matrix_file), sep=';')
    print('\nOD Matrix Created and Exported.\n')

    # 13. (Optional) Plotting
    if plot_bool:
        # Merge OD data with stop coordinates to plot OD lines
        OD_with_from = OD_matrix_By_Day.merge(
            StopMatrix[['matched_bus_stop_1_number', 'bus_stop_1_name', 'bus_stop_1_lat', 'bus_stop_1_lon']],
            left_on='BoardingStop',
            right_on='matched_bus_stop_1_number',
            how='left'
        )

        OD_with_from = OD_with_from.merge(
            StopMatrix[['matched_bus_stop_2_number', 'bus_stop_2_name', 'bus_stop_2_lat', 'bus_stop_2_lon']],
            left_on='Final_AlightingStop',
            right_on='matched_bus_stop_2_number',
            how='left'
        )

        # Convert coordinate strings to float
        OD_with_from['bus_stop_1_lat'] = OD_with_from['bus_stop_1_lat'].str.replace(',', '.').astype(float)
        OD_with_from['bus_stop_1_lon'] = OD_with_from['bus_stop_1_lon'].str.replace(',', '.').astype(float)
        OD_with_from['bus_stop_2_lat'] = OD_with_from['bus_stop_2_lat'].str.replace(',', '.').astype(float)
        OD_with_from['bus_stop_2_lon'] = OD_with_from['bus_stop_2_lon'].str.replace(',', '.').astype(float)

        # Create a LineString geometry for each OD pair
        OD_with_from['geometry'] = OD_with_from.apply(
            lambda row: LineString([
                (row['bus_stop_1_lon'], row['bus_stop_1_lat']),
                (row['bus_stop_2_lon'], row['bus_stop_2_lat'])
            ]),
            axis=1
        )

        # Remove rows where from == to
        OD_with_from = OD_with_from[
            OD_with_from['matched_bus_stop_1_number'] != OD_with_from['matched_bus_stop_2_number']
        ]

        # Create GeoDataFrame
        OD_with_from_gdf = gpd.GeoDataFrame(OD_with_from, geometry='geometry', crs="EPSG:4326")

        # Set line width proportional to the passenger count
        OD_with_from_gdf["linewidth"] = OD_with_from_gdf["count"]

        # Load OSM data (graph, buildings, etc.) from pickle
        with open(os.path.join(mainPath, municipality_name + '_osm_data.pkl'), "rb") as file:
            data = pickle.load(file)

            # Plot OSM layers + OD lines
            plot_osm_data(
                data["G_roads"],
                data["bus_stops"],
                data["buildings"],
                data["waterways"],
                municipality_name,
                OD_with_from_gdf
            )


if __name__ == "__main__":
    # Run the main process
    main(plot_bool=Plot_Check)
    print('Code Finished!')
