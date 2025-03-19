# -*- coding: utf-8 -*-
"""DailyDengueDataProcessing.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1QnpVOHB9yTaXAm0NDbE3wh1Q_zrmg7jP

Importing packages needed
"""

import requests
import pandas as pd
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, Polygon
import folium
import plotly.express as px
import os
import time
from datetime import datetime
import pytz

"""Updating Files in GitHub Repository


"""

# 📌 Step 1: Define paths relative to the GitHub repository
dengue_hotspot_csv = "dengue_hotspot.csv"
active_dengue_csv = "active_dengue.csv"

# 📌 Step 2: Helper function to save DataFrame to CSV
def save_to_csv(df, file_path):
    if df is not None:
        if os.path.exists(file_path):
            df.to_csv(file_path, mode="a", header=False, index=False)  # Append mode
            print(f"Data appended to {file_path}")
        else:
            df.to_csv(file_path, index=False)  # Create new file
        print(f"Data saved to {file_path}")
    else:
        print("No data to save.")

"""API 1 : Fetch UM Location from Idengue.com"""

um_location_url = "https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/findAddressCandidates?SingleLine=Universiti%20Malaya%2C%20Kuala%20Lumpur%2C%20Wilayah%20Persekutuan%20Kuala%20Lumpur%2C%20MYS&f=json&outSR=%7B%22wkid%22%3A102100%7D&outFields=*&magicKey=dHA9MCN0dj00Yjg3MGE5MSNsb2M9Njc4NTA0MTEjbG5nPTExMiNwbD04NTk4Mjk0MCNsYnM9MTQ6NzM1MzMxODMjbG49V29ybGQ%3D&maxLocations=6"

"""API 2 : List of Hotspot"""

hotspot_location_url = "https://sppk.mysa.gov.my/proxy/proxy.php?https://mygis.mysa.gov.my/erica1/rest/services/iDengue/WM_idengue/MapServer/0/query?f=json&where=1%3D1&returnGeometry=true&spatialRel=esriSpatialRelIntersects&outFields=SPWD.AVT_HOTSPOTMINGGUAN.KUMULATIF_KES%2CSPWD.AVT_HOTSPOTMINGGUAN.TEMPOH_WABAK%2CSPWD.AVT_HOTSPOTMINGGUAN.NEGERI%2CSPWD.AVT_HOTSPOTMINGGUAN.DAERAH%2CSPWD.DBO_LOKALITI_POINTS.LOKALITI"

"""API 3 : List of Kawasan Wabak Aktif"""

active_area_url = "https://sppk.mysa.gov.my/proxy/proxy.php?https://mygis.mysa.gov.my/erica1/rest/services/iDengue/WM_idengue/MapServer/4/query?f=json&where=1%3D1&returnGeometry=true&spatialRel=esriSpatialRelIntersects&outFields=SPWD.AVT_WABAK_IDENGUE_NODM.LOKALITI%2CSPWD.AVT_WABAK_IDENGUE_NODM.TOTAL_KES%2CSPWD.AVT_WABAK_IDENGUE_NODM.NEGERI"

"""Defining API Endpoint"""

# Define API endpoints
API_ENDPOINTS = {
    "um_location": um_location_url,
    "hotspot_location_url" : hotspot_location_url,
    "active_area_url" : active_area_url,
}
print(API_ENDPOINTS)

"""Set Timezone to Malaysia (UTC+8)"""

# Set timezone to Malaysia (UTC+8)
malaysia_tz = pytz.timezone("Asia/Kuala_Lumpur")

"""Define functions for processing different API responses"""

def process_api_1(response_json):
    """Process and visualize data for API 1 with interactive hover tooltips."""
    candidates = response_json.get("candidates", [])

    data = [
        {
            "X": c["attributes"].get("X"),
            "Y": c["attributes"].get("Y"),
            "attributes": c["attributes"],
            "location": c["attributes"].get("LongLabel"),
        }
        for c in candidates
    ]

    if not data:
        print("No data available for API 1.")
        return None

    df = pd.DataFrame(data)

    print(f"API 1: Found {len(df)} locations.")
    for idx, row in df.iterrows():
        print(f"Location {idx + 1}: Location -> {row['location']}")

    # Convert attributes dict to string for display
    df["attributes_str"] = df["attributes"].astype(str)

    # Create interactive scatter plot with hover tooltip
    fig = px.scatter(
        df,
        x="X",
        y="Y",
        labels={"X": "Longitude", "Y": "Latitude"},
        title="API 1 - UM Location",
        color_discrete_sequence=["blue"]
    )

    fig.show()

    return df

def process_api_2(response_json, x_target=101.653045, y_target=3.122496, tolerance=0.1):
    """Process and visualize data for API 2 with interactive hover tooltips."""
    features = response_json.get("features", [])

    filtered_data = []

    for feature in features:
        attr = feature["attributes"]
        area = attr.get("SPWD.DBO_LOKALITI_POINTS.LOKALITI", "Unknown")
        state = attr.get("SPWD.AVT_HOTSPOTMINGGUAN.NEGERI", "Unknown")
        days_duration = attr.get("SPWD.AVT_HOTSPOTMINGGUAN.TEMPOH_WABAK", 0)
        total_cases = attr.get("SPWD.AVT_HOTSPOTMINGGUAN.KUMULATIF_KES", 0)

        if (x_target - tolerance <= feature["geometry"]["x"] <= x_target + tolerance) and \
           (y_target - tolerance <= feature["geometry"]["y"] <= y_target + tolerance):
            filtered_data.append({
                "x": feature["geometry"]["x"],
                "y": feature["geometry"]["y"],
                "date": datetime.now(malaysia_tz).strftime('%d/%m/%Y'),  # Malaysia Time
                "area": area,
                "state": state,
                "days_duration": days_duration,
                "total_active_cases": total_cases
            })

    if not filtered_data:
        print("No matching data found within the specified range.")
        return None

    df = pd.DataFrame(filtered_data)

    print(f"API 2: Found {len(df)} dengue hotspot locations.")
    for idx, row in df.iterrows():
        print(f"Hotspot {idx + 1}: Area -> {row['area']}, Cases -> {row['total_active_cases']}")

    # Create interactive scatter plot with hover tooltip
    fig = px.scatter(
        df,
        x="x",
        y="y",
        hover_data=["area", "total_active_cases"],
        labels={"x": "Longitude", "y": "Latitude"},
        title="API 2 - Dengue Hotspots (5KM Radius)",
        color_discrete_sequence=["red"]
    )

    # Add target location as a marker
    fig.add_scatter(
        x=[x_target],
        y=[y_target],
        mode="markers",
        marker=dict(size=10, color="black", symbol="x"),
        name="Target Location"
    )

    fig.show()

    # Save to Google Drive
    save_to_csv(df, dengue_hotspot_csv)

    return df

def calculate_centroid(rings):
    """Calculate the centroid of a polygon given its rings."""
    all_x = [point[0] for ring in rings for point in ring]
    all_y = [point[1] for ring in rings for point in ring]
    return np.mean(all_x), np.mean(all_y)

def process_api_3(response_json, x_target=101.653045, y_target=3.122496, tolerance=0.045):
    """Process and visualize data for API 3, filtering based on polygon centroid."""
    features = response_json.get("features", [])

    filtered_data = []
    for feature in features:
        rings = feature.get("geometry", {}).get("rings", [])
        if not rings:
            continue

        centroid_x, centroid_y = calculate_centroid(rings)

        # Extract relevant attributes
        attributes = feature.get("attributes", {})
        location = attributes.get("SPWD.AVT_WABAK_IDENGUE_NODM.LOKALITI", "null")
        state = attributes.get("SPWD.AVT_WABAK_IDENGUE_NODM.NEGERI", "null")
        total_cases = attributes.get("SPWD.AVT_WABAK_IDENGUE_NODM.TOTAL_KES", 0)

        if (x_target - tolerance <= centroid_x <= x_target + tolerance) and (y_target - tolerance <= centroid_y <= y_target + tolerance):
            filtered_data.append({
                "attributes": feature["attributes"],
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "date": datetime.now(malaysia_tz).strftime('%d/%m/%Y'),  # Malaysia Time
                "location": location,
                "state": state,
                "total_active_cases": total_cases,
            })

    if not filtered_data:
        print("No matching data found within the specified range.")
        return None

    df = pd.DataFrame(filtered_data)

    print(f"API 3: Found {len(df)} active area centroids.")
    for idx, row in df.iterrows():
        print(f"Centroid {idx + 1}: Location: {row['location']}, Total Cases: {row['total_active_cases']}")

    # Convert attributes dict to string for display
    df["attributes_str"] = df["attributes"].astype(str)

    # Create interactive scatter plot with hover tooltip
    fig = px.scatter(
        df,
        x="centroid_x",
        y="centroid_y",
        hover_data=["location", "state", "total_active_cases"],
        labels={"centroid_x": "Longitude", "centroid_y": "Latitude"},
        title="API 3 - Active Area Centroids (5KM Radius)"
    )

    # Add target location as a marker
    fig.add_scatter(
        x=[x_target],
        y=[y_target],
        mode="markers",
        marker=dict(size=10, color="black", symbol="x"),
        name="Target Location"
    )

    fig.show()

    # Drop the unwanted columns
    df = df.drop(columns=["attributes", "attributes_str"])

    # Save to Google Drive
    save_to_csv(df, active_dengue_csv)

    return df

"""Mapping APIs to their respective processing functions"""

PROCESSING_FUNCTIONS = {
    "um_location": process_api_1,
    "hotspot_location_url": process_api_2,
    "active_area_url": process_api_3
}
print(PROCESSING_FUNCTIONS)

"""Define function to fetch and store the data"""

def fetch_and_store(api_name, url):
    """Fetch data from API and store it using the appropriate processing function."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if api_name in PROCESSING_FUNCTIONS:
            df = PROCESSING_FUNCTIONS[api_name](data)
            print(f"Data from {api_name} processed successfully!")
            return df
        else:
            print(f"No processing function defined for {api_name}")
    except Exception as e:
        print(f"Error fetching data from {api_name}: {e}")

"""Run daily data fetching"""

if __name__ == "__main__":
    for api_name, url in API_ENDPOINTS.items():
        print(f"Fetching data from {api_name}...")
        fetch_and_store(api_name, url)
