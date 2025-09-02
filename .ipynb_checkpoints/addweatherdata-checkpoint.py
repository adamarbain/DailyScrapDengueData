import pandas as pd
import numpy as np

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

if __name__ == "__main__":
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
