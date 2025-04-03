import pandas as pd

def preprocess_data():
    file_path = "data/hotel_bookings.csv"  # Ensure the CSV file is in the 'data/' folder
    df = pd.read_csv(file_path)

    # Convert date column
    df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'], errors='coerce')

    # Fill missing values
    df.fillna(0, inplace=True)

    return df
