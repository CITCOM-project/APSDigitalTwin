# APSDigitalTwin: A Digital Twin for the OpenAPS oref0 implementation

APSDigitalTwin (Artificial Pancreas System Digital Twin) uses a predictive model to simulate a person with diabetes with the intervention of OpenAPS's oref0 algorithm. The system learns a model of the user from past obervational data and allows for this model to be run against different scenarios both with and without OpenAPS intervention.

This project was executed with oref0 version 0.7.1, python version 3.9.7 on the operating system Ubuntu 20.04.5 LTS.

## Installation

This system requires OpenAPS oref0 to be installed on the commandline. Please check out the following link for instruction on how to do this: [click here](https://github.com/openaps/oref0)

Install a conda environment for APSDigitalTwin:
```
conda create --force -n aps-digital-twin python=3.9
conda activate aps-digital-twin
pip install -r requirements.txt
```

## Pre-Commit Hooks
We have pre-commit hooks set up to ensure that we don't accidentally commit a file containing a patient id.
The "naughty strings" come from the `constants` directory, so this needs to exist.
Constants are stored in the format `patientID_inx.txt`, where `patientID` is a numerical string representing the openHumans patient ID and `inx` is the index of the trace segment from which the constants were inferred.
For file `constants/patientID_inx.txt`, you are not allowed to commit any file whose name or body contains the string `patientID`.

**CAUTION:** We do not check binary files or compressed folders, so, e.g. .xlsx files would not be checked.
Commit these files at your own risk.

## Data Preparation

This model requires a blood glucose (mmol/L), insulin on board (U), carbohydrates on board (g) and pump output rate (U/h) timeseries at 5 minute intervals to learn the model. This data should be presented in a csv with the following layout:

| bg | iob | cob | rate |
| --- | --- | --- | --- |
| 103 | 0.34 | 5.2 | 1.4 |
| 104 | 0.32 | 5.1 | 0 |


## Model Execution

In `scripts/main.py`, modify `training_data` with a path your own training dataset. You should also update `.env` with the correct path for `profile_path` and `basal_profile_path`.

For windows users, `COMSPEC` should also be updated to point to the exe of the bash command line which has oref0 installed (eg: `\User\GitBash.exe`).

You may also update any scenarios as required. To then run the code:
```
python ./scripts/main.py
```

In each research question python file, the variable `figure_save_path` should be set to a path to save figures.

To execute the research questions, run the following commands:

These scripts can be run with the following commands:
```
python scripts/rq1_model_correctness.py
python scripts/rq2_person_glucose_dynamics.py
python scripts/rq3_openaps_scenarios.py
```

## Output

For `main.py`, after each scenario, the program will display two graphs representing the scenario with no OpenAPS intervention and the scenario with OpenAPS intervention every 5 minutes. Each scenario run will also return `True` or `False` depending if the scenario has less or more deviations outside of a safe blood glucose level.

For each research question script, figures wil be saved to the path represented by `figure_save_path`. Figures generated are specific to the research question in question.
