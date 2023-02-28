# APSDigitalTwin: A Digital Twin for the OpenAPS oref0 implementation

APSDigitalTwin (Artificial Pancreas System Digital Twin) uses a predictive model to simulate a person with diabetes with the intervention of OpenAPS's oref0 algorithm. The system learns a model of the user from past obervational data and allows for this model to be run against different scenarios both with and without OpenAPS intervention.

## Installation

This system requires OpenAPS oref0 to be installed on the commandline. Please check out the following link for instruction on how to do this: [click here](https://github.com/openaps/oref0)

Install a conda environment for APSDigitalTwin:
```
conda create --force -n aps-digital-twin
conda activate aps-digital-twin
pip install -r requirements.txt
```
## Data Preparation

This model requires a blood glucose (mmol/L), insulin on board (U), carbohydrates on board (g) and pump output rate (U/h) timeseries at 5 minute intervals to learn the model. This data should be presented in a csv with the following layout:

| bg | iob | cob | rate |
| --- | --- | --- | --- |
| 103 | 0.34 | 5.2 | 1.4 |
| 104 | 0.32 | 5.1 | 0 |


## Model Execution

In `scripts/main.py`, modify `training_data` with a path your own training dataset and update `profile` and `basal_profile` with paths to the appropriate json files. You may also update any scenarios as required. To then run the code:
```
python ./scripts/main.py
```

## Output

After each scenario, the program will display two graphs representing the scenario with no OpenAPS intervention and the scenario with OpenAPS intervention every 5 minutes. Each scenario run will also return `True` or `False` depending if the scenario has less or more deviations outside of a safe blood glucose level.