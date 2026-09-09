# Wearable Sensor Data Quality Dashboard

An end-to-end Python and Streamlit project for auditing, aligning, and visualising multimodal wearable sensor data collected during controlled stress sessions.

This is a portfolio demonstration using an open research dataset. It is not a clinical device, does not diagnose stress, and should not be used for patient-care decisions.

## Live demo

Deployment link: _to be added after Streamlit Community Cloud deployment_

## Project objectives

- Build a reproducible raw-to-processed data pipeline.
- Validate file structure, sampling frequency, and numeric values.
- Align heart rate, electrodermal activity, and accelerometer signals on a shared time axis.
- Detect observable technical quality issues without relying on manual labels.
- Provide session-level quality summaries for rapid review.
- Communicate results through an interactive Streamlit dashboard.

## Data source

The project uses the **STRESS** recordings from:

Hongn, A., Bosch, F., Prado, L., and Bonomini, P. (2025). *Wearable Device Dataset from Induced Stress and Structured Exercise Sessions* (Version 1.0.1). PhysioNet. https://doi.org/10.13026/he0v-tf17

### Data used in this project

The full PhysioNet resource contains stress, aerobic-exercise, and anaerobic-exercise recordings. To keep the portfolio project focused, this pipeline processes only the controlled stress-session folders and three signals:

| Signal | Meaning | Source rate | Use in this project |
|---|---|---:|---|
| `HR` | Heart rate derived from blood volume pulse | 1 Hz | Cardiovascular trend |
| `EDA` | Electrodermal activity | 4 Hz | Sympathetic-arousal trend |
| `ACC` | Three-axis acceleration in units of 1/64 g | 32 Hz | Movement level and variability |

The first row of each fixed-frequency Empatica file contains the recording start time. The second row contains the sampling frequency. The remaining rows contain sensor observations. Event-button timestamps are stored separately in `tags.csv`.

Participants labelled `Sxx` belong to protocol version V1; participants labelled `fxx` belong to the updated V2 protocol. Source timestamps were de-identified by the dataset authors while maintaining alignment between signals.

## Processing workflow

```text
PhysioNet E4 CSV files
        ↓
Discover stress-session folders
        ↓
Validate schema, timestamps, sample rates, and numeric values
        ↓
Reconstruct a timestamp for every sensor observation
        ↓
Run raw-signal quality checks
        ↓
Aggregate HR, EDA, and ACC to a common one-second grid
        ↓
Outer-join signals by timestamp and measure coverage
        ↓
Run session-level quality checks
        ↓
Write analysis-ready CSV outputs
        ↓
Load outputs into the Streamlit dashboard
```

## Author

Rebecca Zhu — Master of Data Science, University of Auckland
