# Pathfinder Data Tools

Python utilities for processing and visualizing Pathfinder gradiometer data.

## Installation

Install required Python packages:

```bash
pip install pandas matplotlib
```

Or use the provided requirements file:

```bash
pip install -r requirements.txt
```

## Tools

### visualize_data.py

Quick field visualization of survey data. Useful for checking data quality immediately after collection.

**Basic usage:**

```bash
# View all pairs as time series
python visualize_data.py PATH0001.CSV

# View specific pair
python visualize_data.py PATH0001.CSV --pair 1

# Generate spatial map (requires GPS data)
python visualize_data.py PATH0001.CSV --map

# Show gradient distributions
python visualize_data.py PATH0001.CSV --hist

# Print statistics only (no plots)
python visualize_data.py PATH0001.CSV --stats-only
```

**Example output:**

```
Loaded 1234 samples from PATH0001.CSV

============================================================
PATHFINDER DATA STATISTICS
============================================================

GPS Lock: 1150/1234 samples (93.2%)
  Lat range: 51.234567 to 51.235678
  Lon range: 18.345678 to 18.346789

Gradient Statistics (ADC counts):
Pair     Mean       Std Dev    Min        Max
------------------------------------------------
1        225.3      45.2       -120       850
2        218.7      42.8       -105       780
3        210.5      40.1       -98        720
4        223.1      44.5       -115       810

Survey Duration: 15.2 minutes (912 seconds)
Sample Rate: 10.1 Hz (average)
============================================================
```

## Example Data

The `example_data.csv` file contains synthetic data for testing the visualization tools. Use it to verify your Python environment is set up correctly:

```bash
python visualize_data.py example_data.csv
```

## Custom Processing

For more advanced processing (filtering, calibration, export to GIS formats), see the templates in the `processing/` directory (coming soon).

### Basic Python Processing Example

```python
import pandas as pd

# Load data
df = pd.read_csv('PATH0001.CSV')

# Filter to GPS-locked samples only
df_gps = df[(df['lat'] != 0) & (df['lon'] != 0)]

# Apply calibration offset (example)
df_gps['g1_grad_cal'] = df_gps['g1_grad'] - df_gps['g1_grad'].mean()

# Export to GeoJSON for QGIS
import json
features = []
for _, row in df_gps.iterrows():
    features.append({
        'type': 'Feature',
        'geometry': {
            'type': 'Point',
            'coordinates': [row['lon'], row['lat']]
        },
        'properties': {
            'gradient': row['g1_grad_cal']
        }
    })

with open('survey.geojson', 'w') as f:
    json.dump({'type': 'FeatureCollection', 'features': features}, f)
```

## Contributing

Improvements and additional tools welcome! Submit pull requests or open issues on the project repository.
