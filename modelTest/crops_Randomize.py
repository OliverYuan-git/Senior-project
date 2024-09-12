import pandas as pd

filename = 'WinnipegDataset.csv' 
df = pd.read_csv(filename)
count_label_7 = df[df['label'] == 7].shape[0]
print(f"Number of data points labeled as '7': {count_label_7}")

#Randomize
sample_size = 10000
random_sample = df.sample(n=sample_size, random_state=100)
count_label_7_in_sample = random_sample[random_sample['label'] == 7].shape[0]
print(f"Number of '7' labels in the random sample of {sample_size} data points: {count_label_7_in_sample}")
