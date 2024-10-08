import pandas as pd

# Load the CSV file
file_path = 'winequality-white-corrected.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Convert the "quality" column based on the conditions
df['quality'] = df['quality'].apply(lambda x: 0 if x < 8 else 1)

# Save the modified dataframe to a new CSV file
output_file_path = 'winequality-white-modified.csv'  # Replace with desired output file path
df.to_csv(output_file_path, index=False)

print(f"Modified file saved as {output_file_path}")
