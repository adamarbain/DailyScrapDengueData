import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import json

def read_dengue_data(csv_file_path='active_dengue.csv'):
    """
    Read dengue data from CSV file
    
    Args:
        csv_file_path (str): Path to the CSV file containing dengue data
        
    Returns:
        pandas.DataFrame: DataFrame containing dengue data with columns:
            - centroid_x: Longitude coordinate
            - centroid_y: Latitude coordinate  
            - date: Date of the data
            - location: Location description
            - state: State name
            - total_active_cases: Number of active dengue cases
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        
        # Display basic information about the dataset
        print(f"Successfully loaded dengue data from {csv_file_path}")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Total records: {len(df)}")
        print(f"Unique states: {df['state'].nunique()}")
        print(f"Total active cases: {df['total_active_cases'].sum()}")
        
        # Display first few rows
        print("\nFirst 5 rows:")
        print(df.head())
        
        # Display data types
        print("\nData types:")
        print(df.dtypes)
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return None

def get_data_summary(df):
    """
    Get summary statistics of the dengue data
    
    Args:
        df (pandas.DataFrame): DataFrame containing dengue data
        
    Returns:
        dict: Summary statistics
    """
    if df is None:
        return None
        
    summary = {
        'total_records': len(df),
        'date_range': {
            'start': df['date'].min(),
            'end': df['date'].max()
        },
        'states': df['state'].value_counts().to_dict(),
        'total_cases': df['total_active_cases'].sum(),
        'avg_cases_per_location': df['total_active_cases'].mean(),
        'max_cases_single_location': df['total_active_cases'].max(),
        'coordinate_range': {
            'longitude': {'min': df['centroid_x'].min(), 'max': df['centroid_x'].max()},
            'latitude': {'min': df['centroid_y'].min(), 'max': df['centroid_y'].max()}
        }
    }
    
    return summary

def find_historical_records(csv_file_path='active_dengue.csv', sample_size=5):
    """
    Get the first five rows from the CSV file for weather API testing
    
    Args:
        csv_file_path (str): Path to the CSV file
        sample_size (int): Number of records to return (default: 5)
        
    Returns:
        pandas.DataFrame: DataFrame with first five records
    """
    try:
        df = pd.read_csv(csv_file_path)
        
        # Take the first five rows
        sample_df = df.head(sample_size).copy()
        
        print(f"Using first {len(sample_df)} records for preview:")
        print(sample_df[['centroid_x', 'centroid_y', 'date', 'location', 'state']].to_string())
        
        return sample_df
        
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return None

def add_missing_weather_data(csv_file_path='active_dengue.csv', delay_between_requests=1):
    """
    Add weather data for rows that are missing humidity, temperature, or rainfall data
    
    Args:
        csv_file_path (str): Path to the CSV file
        delay_between_requests (float): Delay in seconds between API requests
        
    Returns:
        pandas.DataFrame: Enhanced DataFrame with missing weather data filled
    """
    try:
        # Read the CSV file
        print(f"Reading dengue data from {csv_file_path}...")
        df = pd.read_csv(csv_file_path)
        
        # Find rows with missing weather data
        missing_mask = df['humidity'].isna() | df['temperature'].isna() | df['rainfall'].isna()
        missing_rows = df[missing_mask]
        
        if len(missing_rows) == 0:
            print("No missing weather data found. All rows already have weather data.")
            return df
        
        print(f"Found {len(missing_rows)} rows with missing weather data")
        
        # Get unique combinations of coordinates and dates for missing data
        unique_combinations = missing_rows[['centroid_x', 'centroid_y', 'date']].drop_duplicates()
        total_combinations = len(unique_combinations)
        
        print(f"Found {total_combinations} unique location-date combinations with missing data")
        print("Fetching weather data from Open Meteo APIs...")
        
        # Create a dictionary to store weather data for each unique combination
        weather_cache = {}
        
        for idx, row in unique_combinations.iterrows():
            lon, lat, date = row['centroid_x'], row['centroid_y'], row['date']
            
            # Create a cache key
            cache_key = f"{lon}_{lat}_{date}"
            
            if cache_key not in weather_cache:
                # Check if date is in the future or today
                try:
                    date_obj = datetime.strptime(date, '%d/%m/%Y')
                    today = datetime.now().date()
                    
                    # Use forecast API for today, yesterday, and future dates
                    # Archive API often has delays for recent dates
                    if date_obj.date() >= today - timedelta(days=1):
                        print(f"Fetching forecast data for date {date} at location ({lon:.4f}, {lat:.4f})...")
                        weather_data = fetch_weather_data_forecast(lat, lon, date)
                    else:
                        print(f"Fetching archive data for historical date {date} at location ({lon:.4f}, {lat:.4f})...")
                        weather_data = fetch_weather_data(lat, lon, date)
                        
                except Exception as e:
                    print(f"Date parsing failed for {date}: {str(e)}")
                    print(f"Fetching forecast data for date {date} at location ({lon:.4f}, {lat:.4f})...")
                    weather_data = fetch_weather_data_forecast(lat, lon, date)
                
                weather_cache[cache_key] = weather_data
                
                # Add delay to avoid rate limiting
                time.sleep(delay_between_requests)
            else:
                print(f"Using cached weather data for location ({lon:.4f}, {lat:.4f}) on {date}")
        
        # Apply weather data to missing rows
        print("Applying weather data to missing rows...")
        for idx, row in missing_rows.iterrows():
            lon, lat, date = row['centroid_x'], row['centroid_y'], row['date']
            cache_key = f"{lon}_{lat}_{date}"
            
            if cache_key in weather_cache:
                weather_data = weather_cache[cache_key]
                
                # Only update missing values
                if pd.isna(df.at[idx, 'humidity']):
                    df.at[idx, 'humidity'] = weather_data['humidity']
                if pd.isna(df.at[idx, 'temperature']):
                    df.at[idx, 'temperature'] = weather_data['temperature']
                if pd.isna(df.at[idx, 'rainfall']):
                    df.at[idx, 'rainfall'] = weather_data['rainfall']
        
        # Save the enhanced CSV file
        df.to_csv(csv_file_path, index=False)
        print(f"Enhanced CSV file saved to: {csv_file_path}")
        
        # Display summary of added weather data
        print("\nWeather Data Summary:")
        print(f"Records with humidity data: {df['humidity'].notna().sum()}")
        print(f"Records with temperature data: {df['temperature'].notna().sum()}")
        print(f"Records with rainfall data: {df['rainfall'].notna().sum()}")
        
        if df['humidity'].notna().any():
            print(f"Average humidity: {df['humidity'].mean():.2f}%")
        if df['temperature'].notna().any():
            print(f"Average temperature: {df['temperature'].mean():.2f}°C")
        if df['rainfall'].notna().any():
            print(f"Total rainfall: {df['rainfall'].sum():.2f}mm")
        
        return df
        
    except Exception as e:
        print(f"Error adding missing weather data: {str(e)}")
        return None

def fetch_weather_data_forecast(latitude, longitude, date, timezone="Asia/Singapore"):
    """
    Fetch weather data from Open Meteo Forecast API for future dates
    
    Args:
        latitude (float): Latitude coordinate
        longitude (float): Longitude coordinate
        date (str): Date in format 'DD/MM/YYYY'
        timezone (str): Timezone for the API request
        
    Returns:
        dict: Weather data containing humidity, temperature, and rainfall
    """
    try:
        # Convert date from DD/MM/YYYY to YYYY-MM-DD format
        date_obj = datetime.strptime(date, '%d/%m/%Y')
        formatted_date = date_obj.strftime('%Y-%m-%d')
        
        # Construct Forecast API URL
        base_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'daily': 'precipitation_sum,temperature_2m_max,temperature_2m_min',
            'hourly': 'relative_humidity_2m',
            'timezone': timezone,
            'start_date': formatted_date,
            'end_date': formatted_date
        }
        
        # Make API request
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract weather data
        weather_data = {
            'humidity': None,
            'temperature': None,
            'rainfall': None
        }
        
        # Calculate average humidity from hourly data
        if 'hourly' in data and 'relative_humidity_2m' in data['hourly']:
            humidity_values = data['hourly']['relative_humidity_2m']
            if humidity_values and all(v is not None for v in humidity_values):
                weather_data['humidity'] = sum(humidity_values) / len(humidity_values)
        
        # Extract daily temperature (average of max and min) and rainfall
        if 'daily' in data:
            if 'temperature_2m_max' in data['daily'] and 'temperature_2m_min' in data['daily']:
                temp_max = data['daily']['temperature_2m_max'][0] if data['daily']['temperature_2m_max'] else None
                temp_min = data['daily']['temperature_2m_min'][0] if data['daily']['temperature_2m_min'] else None
                if temp_max is not None and temp_min is not None:
                    weather_data['temperature'] = (temp_max + temp_min) / 2
            
            if 'precipitation_sum' in data['daily'] and data['daily']['precipitation_sum']:
                weather_data['rainfall'] = data['daily']['precipitation_sum'][0]
        
        return weather_data
        
    except requests.exceptions.RequestException as e:
        print(f"Forecast API request failed for lat={latitude}, lon={longitude}, date={date}: {str(e)}")
        return {'humidity': None, 'temperature': None, 'rainfall': None}
    except Exception as e:
        print(f"Error processing forecast weather data for lat={latitude}, lon={longitude}, date={date}: {str(e)}")
        return {'humidity': None, 'temperature': None, 'rainfall': None}

def fetch_weather_data(latitude, longitude, date, timezone="Asia/Singapore"):
    """
    Fetch weather data from Open Meteo API for a specific location and date
    
    Args:
        latitude (float): Latitude coordinate
        longitude (float): Longitude coordinate
        date (str): Date in format 'DD/MM/YYYY'
        timezone (str): Timezone for the API request
        
    Returns:
        dict: Weather data containing humidity, temperature, and rainfall
    """
    try:
        # Convert date from DD/MM/YYYY to YYYY-MM-DD format
        date_obj = datetime.strptime(date, '%d/%m/%Y')
        formatted_date = date_obj.strftime('%Y-%m-%d')
        
        # Check if date is in the future (Open Meteo archive only has historical data)
        today = datetime.now().date()
        if date_obj.date() > today:
            print(f"Warning: Date {date} is in the future. Open Meteo archive API only provides historical data.")
            return {'humidity': None, 'temperature': None, 'rainfall': None}
        elif date_obj.date() == today:
            print(f"Warning: Date {date} is today. Weather data might be incomplete or unavailable.")
        
        # Construct API URL
        base_url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'start_date': formatted_date,
            'end_date': formatted_date,
            'daily': 'temperature_2m_mean,rain_sum',
            'hourly': 'relative_humidity_2m',
            'timezone': timezone
        }
        
        # Make API request
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract weather data
        weather_data = {
            'humidity': None,
            'temperature': None,
            'rainfall': None
        }
        
        # Calculate average humidity from hourly data
        if 'hourly' in data and 'relative_humidity_2m' in data['hourly']:
            humidity_values = data['hourly']['relative_humidity_2m']
            if humidity_values and all(v is not None for v in humidity_values):
                weather_data['humidity'] = sum(humidity_values) / len(humidity_values)
        
        # Extract daily temperature and rainfall
        if 'daily' in data:
            if 'temperature_2m_mean' in data['daily'] and data['daily']['temperature_2m_mean']:
                weather_data['temperature'] = data['daily']['temperature_2m_mean'][0]
            
            if 'rain_sum' in data['daily'] and data['daily']['rain_sum']:
                weather_data['rainfall'] = data['daily']['rain_sum'][0]
        
        return weather_data
        
    except requests.exceptions.RequestException as e:
        print(f"API request failed for lat={latitude}, lon={longitude}, date={date}: {str(e)}")
        return {'humidity': None, 'temperature': None, 'rainfall': None}
    except Exception as e:
        print(f"Error processing weather data for lat={latitude}, lon={longitude}, date={date}: {str(e)}")
        return {'humidity': None, 'temperature': None, 'rainfall': None}

def add_weather_data_to_csv(csv_file_path='active_dengue.csv', output_file_path=None, delay_between_requests=1):
    """
    Add weather data columns to the dengue CSV file
    
    Args:
        csv_file_path (str): Path to the input CSV file
        output_file_path (str): Path to save the enhanced CSV file (default: overwrite original)
        delay_between_requests (float): Delay in seconds between API requests to avoid rate limiting
        
    Returns:
        pandas.DataFrame: Enhanced DataFrame with weather data
    """
    try:
        # Read the original CSV file
        print(f"Reading dengue data from {csv_file_path}...")
        df = pd.read_csv(csv_file_path)
        
        # Initialize new columns
        df['humidity'] = None
        df['temperature'] = None
        df['rainfall'] = None
        
        # Get unique combinations of coordinates and dates to minimize API calls
        unique_combinations = df[['centroid_x', 'centroid_y', 'date']].drop_duplicates()
        total_combinations = len(unique_combinations)
        
        print(f"Found {total_combinations} unique location-date combinations")
        print("Fetching weather data from Open Meteo API...")
        
        # Create a dictionary to store weather data for each unique combination
        weather_cache = {}
        
        for idx, row in unique_combinations.iterrows():
            lon, lat, date = row['centroid_x'], row['centroid_y'], row['date']
            
            # Create a cache key
            cache_key = f"{lon}_{lat}_{date}"
            
            if cache_key not in weather_cache:
                # Check if date is in the future or today
                try:
                    date_obj = datetime.strptime(date, '%d/%m/%Y')
                    today = datetime.now().date()
                    if date_obj.date() > today:
                        print(f"Skipping future date {date} for location ({lon:.4f}, {lat:.4f})...")
                        weather_cache[cache_key] = {'humidity': None, 'temperature': None, 'rainfall': None}
                        continue
                    elif date_obj.date() == today:
                        print(f"Warning: Date {date} is today - weather data might be incomplete for location ({lon:.4f}, {lat:.4f})...")
                except:
                    pass  # If date parsing fails, try to fetch anyway
                
                print(f"Fetching weather data for location ({lon:.4f}, {lat:.4f}) on {date}...")
                
                # Fetch weather data
                weather_data = fetch_weather_data(lat, lon, date)
                weather_cache[cache_key] = weather_data
                
                # Add delay to avoid rate limiting
                time.sleep(delay_between_requests)
            else:
                print(f"Using cached weather data for location ({lon:.4f}, {lat:.4f}) on {date}")
        
        # Apply weather data to the DataFrame
        print("Applying weather data to all rows...")
        for idx, row in df.iterrows():
            lon, lat, date = row['centroid_x'], row['centroid_y'], row['date']
            cache_key = f"{lon}_{lat}_{date}"
            
            if cache_key in weather_cache:
                weather_data = weather_cache[cache_key]
                df.at[idx, 'humidity'] = weather_data['humidity']
                df.at[idx, 'temperature'] = weather_data['temperature']
                df.at[idx, 'rainfall'] = weather_data['rainfall']
        
        # Save the enhanced CSV file
        if output_file_path is None:
            output_file_path = csv_file_path
        
        df.to_csv(output_file_path, index=False)
        print(f"Enhanced CSV file saved to: {output_file_path}")
        
        # Display summary of added weather data
        print("\nWeather Data Summary:")
        print(f"Records with humidity data: {df['humidity'].notna().sum()}")
        print(f"Records with temperature data: {df['temperature'].notna().sum()}")
        print(f"Records with rainfall data: {df['rainfall'].notna().sum()}")
        
        if df['humidity'].notna().any():
            print(f"Average humidity: {df['humidity'].mean():.2f}%")
        if df['temperature'].notna().any():
            print(f"Average temperature: {df['temperature'].mean():.2f}°C")
        if df['rainfall'].notna().any():
            print(f"Total rainfall: {df['rainfall'].sum():.2f}mm")
        
        return df
        
    except Exception as e:
        print(f"Error adding weather data: {str(e)}")
        return None

def preview_weather_integration(csv_file_path='active_dengue.csv', sample_size=5):
    """
    Preview the weather data integration process with a small sample
    
    Args:
        csv_file_path (str): Path to the CSV file
        sample_size (int): Number of sample records to process
    """
    try:
        # Find historical records (not future dates)
        sample_df = find_historical_records(csv_file_path, sample_size)
        
        if sample_df is None:
            print("Cannot proceed with preview - no historical records found.")
            return None
        
        print(f"\nPreview: Processing {len(sample_df)} historical records...")
        
        # Initialize weather columns
        sample_df['humidity'] = None
        sample_df['temperature'] = None
        sample_df['rainfall'] = None
        
        # Process each sample record
        for idx, row in sample_df.iterrows():
            lon, lat, date = row['centroid_x'], row['centroid_y'], row['date']
            
            print(f"\nFetching weather for: {row['location'][:50]}... on {date}")
            weather_data = fetch_weather_data(lat, lon, date)
            
            sample_df.at[idx, 'humidity'] = weather_data['humidity']
            sample_df.at[idx, 'temperature'] = weather_data['temperature']
            sample_df.at[idx, 'rainfall'] = weather_data['rainfall']
            
            # Format weather data safely, handling None values
            humidity_str = f"{weather_data['humidity']:.2f}%" if weather_data['humidity'] is not None else "N/A"
            temp_str = f"{weather_data['temperature']:.2f}°C" if weather_data['temperature'] is not None else "N/A"
            rain_str = f"{weather_data['rainfall']:.2f}mm" if weather_data['rainfall'] is not None else "N/A"
            
            print(f"Weather data: Humidity={humidity_str}, "
                  f"Temperature={temp_str}, "
                  f"Rainfall={rain_str}")
            
            time.sleep(1)  # Delay between requests
        
        print("\nSample results:")
        print(sample_df[['location', 'date', 'humidity', 'temperature', 'rainfall']].to_string())
        
        return sample_df
        
    except Exception as e:
        print(f"Error in preview: {str(e)}")
        return None

if __name__ == "__main__":
    print("="*60)
    print("DENGUE DATA WEATHER INTEGRATION TOOL")
    print("="*60)
    
    # Read the dengue data
    dengue_data = read_dengue_data('active_dengue.csv')
    
    if dengue_data is not None:
        # Get and display summary
        summary = get_data_summary(dengue_data)
        print("\n" + "="*50)
        print("DATA SUMMARY")
        print("="*50)
        print(f"Total records: {summary['total_records']}")
        print(f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        print(f"Total active cases: {summary['total_cases']}")
        print(f"Average cases per location: {summary['avg_cases_per_location']:.2f}")
        print(f"Maximum cases in single location: {summary['max_cases_single_location']}")
        print(f"Number of states: {len(summary['states'])}")
        print("\nCases by state:")
        for state, count in summary['states'].items():
            print(f"  {state}: {count} locations")
        
        print("\n" + "="*50)
        print("WEATHER INTEGRATION OPTIONS")
        print("="*50)
        print("1. Preview weather integration (5 historical records)")
        print("2. Add weather data to full dataset")
        print("3. Add missing weather data (for rows with empty weather columns)")
        print("4. Exit")
        print("\nNote: Uses Archive API for historical data and Forecast API for future dates.")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            print("\nRunning preview with 5 sample records...")
            preview_weather_integration('active_dengue.csv', sample_size=5)
            
        elif choice == "2":
            print("\nStarting full weather data integration...")
            print("This may take a while depending on the number of unique locations and dates.")
            confirm = input("Are you sure you want to proceed? (y/n): ").strip().lower()
            
            if confirm == 'y':
                # Create backup of original file
                backup_file = f"active_dengue_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                dengue_data.to_csv(backup_file, index=False)
                print(f"Backup created: {backup_file}")
                
                # Add weather data
                enhanced_data = add_weather_data_to_csv('active_dengue.csv')
                
                if enhanced_data is not None:
                    print("\nWeather data integration completed successfully!")
                    print("New columns added: humidity, temperature, rainfall")
                else:
                    print("Weather data integration failed.")
            else:
                print("Operation cancelled.")
                
        elif choice == "3":
            print("\nAdding missing weather data...")
            print("This will fetch weather data only for rows with empty weather columns.")
            confirm = input("Are you sure you want to proceed? (y/n): ").strip().lower()
            
            if confirm == 'y':
                enhanced_data = add_missing_weather_data('active_dengue.csv')
                
                if enhanced_data is not None:
                    print("\nMissing weather data integration completed successfully!")
                else:
                    print("Missing weather data integration failed.")
            else:
                print("Operation cancelled.")
                
        elif choice == "4":
            print("Exiting...")
        else:
            print("Invalid choice. Please run the script again.")
