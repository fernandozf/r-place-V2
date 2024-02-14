# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 13:08:51 2024

@author: alexa
"""

import glob
import os
import re
import pandas as pd
import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq

root_folder_path = 'C:/Users/alexa/OneDrive - Cal Poly/Cloud Computing/rplace_unzipped'
parquet_file = 'combined.parquet'

csv_files = glob.glob(os.path.join(root_folder_path, '**/*.csv'), recursive=True)

# Define the function to extract X and Y coordinates
def extract_coordinates(coord):
    if '{' in coord:
        match = re.search(r'X: (-?\d+), Y: (-?\d+)', coord)
        if match:
            return int(match.group(1)), int(match.group(2))
    else:
        try:
            x, y = coord.split(',')
            return int(x), int(y)
        except ValueError:
            pass
    return None, None

# Define the final schema, including the new X and Y fields
final_schema = pa.schema([
    pa.field('timestamp', pa.timestamp('ms', tz='UTC')),
    pa.field('user', pa.string()),
    pa.field('X', pa.int32()),
    pa.field('Y', pa.int32()),
    pa.field('pixel_color', pa.string())
])

writer = pq.ParquetWriter(parquet_file, final_schema, compression='snappy')

parse_options = csv.ParseOptions(newlines_in_values=True)
convert_options = csv.ConvertOptions(column_types={'timestamp': pa.string(), 'user': pa.string(), 'coordinate': pa.string(), 'pixel_color': pa.string()})

for file in csv_files:
    print("reading", file)
    table = csv.read_csv(file, parse_options=parse_options, convert_options=convert_options)
    
    # Convert 'timestamp' column using flexible parsing
    timestamps = pd.to_datetime(table.column('timestamp').to_pandas(), utc=True, format='mixed')
    timestamps_arrow = pa.array(timestamps, type=pa.timestamp('ms', tz='UTC'))
    
    # Process 'coordinate' column to extract X and Y
    coordinates_series = table.column('coordinate').to_pandas()
    xy = coordinates_series.apply(extract_coordinates)
    x_list, y_list = zip(*xy)  # This creates two lists of X and Y values
    x_array = pa.array(x_list, type=pa.int32())
    y_array = pa.array(y_list, type=pa.int32())
    
    # Reconstruct the table according to the final schema
    new_table = pa.Table.from_arrays([
        timestamps_arrow,
        table.column('user'),
        x_array,
        y_array,
        table.column('pixel_color')
    ], schema=final_schema)
    
    print("writing", file)
    writer.write_table(new_table)

writer.close()

print(f"Combined CSVs to {parquet_file}")

