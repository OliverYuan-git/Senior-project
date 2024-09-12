import pandas as pd

input_filename = 'WinnipegDataset.txt' 
output_filename = 'WinnipegDataset.csv' 
delimiter = ',' 
chunksize = 10**6 

with pd.read_csv(input_filename, delimiter=delimiter, chunksize=chunksize) as reader:
    for i, chunk in enumerate(reader):
        chunk.to_csv(output_filename, mode='a', index=False, header=i == 0)
print(f"The file has been successfully converted to {output_filename}.")
