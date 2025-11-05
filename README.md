# Predictive-Maintenance-PDM-of-Diesel-Engine

This project focuses on Predictive Maintenance (PdM) of a diesel engine, with the goal of predicting the Remaining Useful Life (RUL) using Long Short-Term Memory (LSTM) deep learning models.

Due to the scarcity of real-world fault data, a physics-informed degradation model was developed in Simulink to generate synthetic time-series datasets. The degradation is modeled as a gradual increase in friction within a critical engine component. The resulting dataset is then used to train an LSTM model for RUL prediction.
