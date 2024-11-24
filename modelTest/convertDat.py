import pandas as pd

# Load the CSV file
csv_file_path = 'data\cgan_white_wine_5.csv'
data = pd.read_csv(csv_file_path)

# Define the .dat file path
dat_file_path = 'cgan_white_5.dat'

# Save to .dat file with space-separated values
data.to_csv(dat_file_path, sep=' ', index=False, header=True)

print(f"File has been successfully converted to {dat_file_path}")
