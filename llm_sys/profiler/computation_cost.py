import pandas as pd

# Read the sorted CSV file
df = pd.read_csv("merged_sorted.csv")
df = df.iloc[6:].reset_index(drop=True)

# Initialize lists to store communication and computation times
communication_costs = []
computation_costs = []

# Iterate through the rows to calculate costs
for i in range(1, len(df)):
    prev_row = df.iloc[i - 1]
    curr_row = df.iloc[i]
    
    # Calculate time difference
    time_diff = curr_row['time_stamp'] - prev_row['time_stamp']
    
    # Communication cost: specific transitions
    if (prev_row['source'] == '1' and curr_row['source'] == '2') or \
       (prev_row['source'] == '2' and curr_row['source'] == 'master') or \
       (prev_row['source'] == 'master' and curr_row['source'] == '1'):
        communication_costs.append(time_diff)
    
    # Computation cost: same source
    elif (prev_row['source'] == '1' and curr_row['source'] == '1') or \
         (prev_row['source'] == '2' and curr_row['source'] == '2'):
        computation_costs.append(time_diff)

# Calculate averages
average_communication_cost = sum(communication_costs) / len(communication_costs) if communication_costs else 0
average_computation_cost = sum(computation_costs) / len(computation_costs) if computation_costs else 0

# Print the results
print(f"Average Communication Cost: {average_communication_cost}")
print(f"Average Computation Cost: {average_computation_cost}")
