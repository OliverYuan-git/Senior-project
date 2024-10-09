from LDA import generate_2_cluster_data 
import csv

num_of_dimension = 2

X,y = generate_2_cluster_data(n=400, d=num_of_dimension, d_max=8)
header=['Quality']

for i in range(0, num_of_dimension):
    header.append(f"feature {i}")

with open('2-cluster.csv', mode='w', newline='') as file:
    file = csv.writer(file)

    file.writerow(header)

    for i in range(0, len(y)):
        data = [y[i]]
        for j in range(0, len(X[i])):
            data.append(X[i][j])
        file.writerow(data)