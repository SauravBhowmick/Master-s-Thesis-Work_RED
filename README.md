# Masters Thesis - Electrical Transient Detection & Classification

## Overview
This repository contains the code and analysis for a Master's thesis focused on detecting and classifying electrical transients in power systems using machine learning techniques. The project analyzes COMTRADE electrical measurement data to identify transient events and classify them into different categories.

## Author
**Saurav Bhowmick**

## Project Description
The project implements a comprehensive pipeline for:
- Loading and processing COMTRADE electrical measurement files
- Analyzing voltage and current waveforms
- Detecting transient events using curve fitting and FFT analysis
- Classifying events using ensemble machine learning models
- Forecasting voltage profiles using LSTM neural networks

## Key Features

### 1. COMTRADE Data Processing
- Loads `.cfg` and `.dat` files from COMTRADE format
- Extracts voltage (U1n, U2n, U3n, U12) and current (L1, L2, L3) measurements
- Converts data to JSON and CSV formats for further analysis
- Generates time-stamped datasets with microsecond precision

### 2. Signal Analysis
- **Cross-Correlation Analysis**: Identifies similar waveform patterns
- **FFT Segmentation**: Performs frequency analysis over time windows
- **Curve Fitting**: Fits sinusoidal models to detect deviations from normal operation
- **Heatmap Visualization**: Displays frequency-time analysis

### 3. Transient Detection
- Identifies transient events by analyzing deviations from fitted sinusoidal curves
- Uses confidence intervals to classify points as normal (0) or transient (1)
- Implements threshold-based detection with adjustable sensitivity

### 4. Machine Learning Classification

#### Multi-Class Classification (4 Classes)
- **Class 0**: Normal operation
- **Class 1**: Transient events (t)
- **Class 2**: Error events (e)
- **Class 3**: Combined transient+error events (te)

#### Binary Classification (2 Classes)
- **Class 0**: Normal operation
- **Class 1**: Any anomaly (transient/error)

#### Models Implemented
- **Balanced Random Forest Classifier**
- **Easy Ensemble Classifier**
- Grid search for hyperparameter optimization
- Feature importance analysis

### 5. Feature Engineering
- RMS values for voltage and current across all three phases
- Total Harmonic Distortion (THD) metrics
- Neutral current measurements
- Temporal features: Hour of day, Day of week
- AZEP (Additional Zero Energy Point) indicators

### 6. LSTM Voltage Forecasting
- Generates synthetic voltage profiles using LSTM neural networks
- Predicts future voltage values based on historical patterns
- Evaluates prediction accuracy using Mean Squared Error (MSE)

## Requirements

```python
pandas
numpy
matplotlib
seaborn
statsmodels
scikit-learn
imbalanced-learn
tensorflow/keras
comtrade
scipy
plotly
```

## Installation

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn imbalanced-learn tensorflow comtrade scipy plotly
```

## Usage

### 1. Load COMTRADE Files
```python
rec = comtrade.load("path/to/file.cfg", "path/to/file.dat")
```

### 2. Extract and Visualize Data
```python
data = np.array(rec.analog[phase+5][start_time:stop_time])
time = np.array(rec.time[start_time:stop_time])
```

### 3. Perform FFT Analysis
```python
segments = segment_and_fft(time, data, segment_duration=0.01)
```

### 4. Train Classification Model
```python
model = BalancedRandomForestClassifier(n_estimators=300, max_depth=7)
model.fit(X_train_scaled, y_train)
```

### 5. Generate Voltage Forecast
```python
synthetic_profile = generate_synthetic_profile(model, seed_sequence, length, scaler)
```

## Data Structure

### Input Files
- **COMTRADE Files**: `.cfg` and `.dat` files containing electrical measurements
- **CSV Files**: Processed RMS values from Schneider UMG801 meters
  - `01_30d_schneider_umg801.csv` through `07_30d_schneider_umg801.csv`
- **Transient Events**: `transient_events_schneider_umg801.csv`

### Generated Outputs
- JSON files with complete time-series data
- CSV files with segmented FFT results
- Heatmaps showing frequency-time analysis
- Confusion matrices for model evaluation
- Feature importance plots

## Model Performance

### Multi-Class Classification Results
- **Balanced Random Forest**: Optimized with GridSearchCV
- **Easy Ensemble Classifier**: Alternative ensemble approach
- Metrics: Precision, Recall, F1-Score, ROC-AUC, MCC

### Binary Classification Results
- Higher accuracy for binary classification tasks
- Excellent performance in detecting any anomaly vs. normal operation

### Key Performance Indicators
- ROC-AUC scores across different model configurations
- Matthews Correlation Coefficient (MCC)
- Per-class precision, recall, and F1-scores
- Confusion matrices for detailed error analysis

## Feature Importance
The analysis identifies the most important features for transient detection:
- Neutral current (IN)
- AZEP indicators
- Phase voltages and currents
- THD metrics
- Temporal features (hour, day of week)

## Visualizations
- Voltage and current waveforms with dual y-axes
- FFT heatmaps showing frequency evolution over time
- Correlation heatmaps for feature relationships
- Confusion matrices with color-coded performance
- Feature importance bar charts
- Original vs. fitted sinusoidal curves with confidence intervals

## Future Work
- Voltage forecasting improvements with increased epochs
- Integration of additional electrical parameters
- Real-time transient detection system
- Extended classification for more event types
- Deep learning approaches for pattern recognition

## Notes
- The code filters out data from May and December 13 for data quality reasons
- Missing values are handled through dropping or imputation depending on the analysis phase
- GridSearchCV is used extensively for hyperparameter optimization
- The analysis uses stratified splitting to maintain class balance

## Contributing
This is a master's thesis project. For questions or collaboration inquiries, please contact the author.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
This project is part of academic research. Please cite appropriately if using this code for research purposes.

### MIT License Summary
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use
- ❌ Liability
- ❌ Warranty

## Citation
If you use this code in your research, please cite:

```bibtex
@mastersthesis{bhowmick2025power,
  author = {Bhowmick, Saurav},
  title = {Analysis and Prediction of Power Quality Anomalies in Power Electronic Dominated Industrial Systems},
  school = {Hochschule Offenburg},
  year = {2025},
  address = {Offenburg, Germany},
  pages = {103},
  url = {https://opus.hs-offenburg.de/10381},
  note = {Master's Thesis, Faculty of Electrical Engineering, Medical Technology and Computer Science (EMI) and Faculty of Mechanical and Process Engineering (M+V)}
}
```

**Full Thesis**: [https://opus.hs-offenburg.de/frontdoor/index/index/docId/10381](https://opus.hs-offenburg.de/frontdoor/index/index/docId/10381)

**Advisors**: Jörg Bausch, Uchenna Johnpaul Aniekwensi

## Acknowledgments
Special thanks to the supervisors and Hochschule Offenburg for providing access to the electrical measurement data and computational resources.

---

**Contact**: For questions about the implementation or methodology, please open an issue in this repository.
