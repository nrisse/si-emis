# Sea Ice Microwave Emissivity Repository (`si-emis`)

This repository contains the code associated with the manuscript titled, 
*Assessing the Sea Ice Microwave Emissivity up to Submillimeter Waves from 
Airborne and Satellite Observations* by Risse et al. (2024), currently under 
submission to *The Cryosphere*.

## Script Execution Order

Follow the outlined order for the scripts to generate the figures.

### Creating Airborne Emissivity Data

1. Run `data_preparation/footprint.py`
2. Run `data_preparation/timeshifts.py`
3. Run `data_preparation/radiance.py`

### Generating Atmospheric Profiles, Camera Plots, etc.

1. Run `data_preparation/profile.py`
2. Run `data_preparation/camera.py`
3. Run `data_preparation/imagery.py`
4. Run `data_preparation/satellite.py`

### Running PAMTRA Simulation and Emissivity Calculation

1. Run `run setupyamls.py`
2. Execute `run_pamtra_simulation.sh aircraft`
3. Execute `run_pamtra_simulation.sh satellite`
4. Run `retrieval/emissivity.py`

### Combining Aircraft and Satellite Data

1. Run `data_preparation/airsat.py`

### Data Analysis, Figures, and Tables

- Analysis: `analysis/*.py`
- Figures: `figures/*.py`
- Tables: `tables/*.py`
