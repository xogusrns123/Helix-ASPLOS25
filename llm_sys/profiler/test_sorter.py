import pandas as pd

# Read the CSV files
master_df = pd.read_csv("test_master.csv")
test_1_df = pd.read_csv("test_1.csv")
test_2_df = pd.read_csv("test_2.csv")

# Add source column to each DataFrame
master_df['source'] = 'master'
test_1_df['source'] = '1'
test_2_df['source'] = '2'

# Adjust time_stamp for test_1.csv and test_2.csv based on the provided offsets
test_1_df['time_stamp'] += 0.015
test_2_df['time_stamp'] -= 0.008

# Concatenate the DataFrames
combined_df = pd.concat([master_df, test_1_df, test_2_df])

# Drop the 'this_iter_processed' column if it exists
combined_df = combined_df.drop(columns=['this_iter_processed'], errors='ignore')

# Sort by time_stamp
sorted_df = combined_df.sort_values(by='time_stamp').reset_index(drop=True)

# Save the sorted DataFrame to a new CSV file
sorted_df.to_csv("merged_sorted.csv", index=False)

print("Merged and sorted data saved to 'merged_sorted.csv'")