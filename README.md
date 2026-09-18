# icond2
IconD2km RUC-EPC 14hr 20x member ensemble, updates roughly 30-35min after hourly update time. Grib files sourced from DWD. Slow launch, unoptimised, formatting bugs when auto updates. Made with ChatGPT with python since my programming experience is limited.

ChatGPT comments for implementing to dashboard:

ICON-D2 RUC-EPS Meteogram
Python meteogram for the DWD ICON-D2 RUC-EPS ensemble.
The script downloads forecast data directly from DWD and plots:

20 individual EPS ensemble members
Ensemble average (AVG)
Operational ICON-D2 run (OPER)
Automatic detection of new model runs
Automatic retrieval of newly available forecast hours
Hover information for individual forecast hours
Requirements
Python 3 with:
pip install requests numpy matplotlib eccodes
Run
python3 meteogram.py
Integration recommendations
For web dashboards, the data retrieval and GRIB extraction logic should be separated from the Matplotlib interface.
Use the script as a reference for:

Detecting the latest DWD ICON-D2 RUC-EPS run
Detecting which forecast hours are available
Downloading and caching GRIB files
Extracting temperature values from the GRIB data
Handling the 20 EPS members and operational run
For a dashboard implementation, expose the extracted data through the dashboard's existing backend/API and render it using the dashboard's charting system rather than embedding the desktop Matplotlib window.
The configured locations currently use native ICON-D2 grid-point indices, which should be preserved or replaced with an appropriate location-to-grid mapping when adding new locations.


