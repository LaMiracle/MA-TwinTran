# Mechanism-aiding Multivariate Timeseries Forecast Toolkit

A Python toolkit for transformer-based in-flight parameter forecasting, including model training, cross-validation, and performance visualization.

## Project Structure

```text
.
├── models/
│   ├── improve_model_15.pt    # Saved PyTorch model weights (Second-phase mechanism-aiding model)
│   └── model_15.pt            # Saved PyTorch model weights base (First-phase data-driven model)
├── utils/
|   ├── earlystop.py           # Early stopping utility for model training
|   ├── PE.py                  # Positional Encoding utilities
|   ├── plotter.py             # Data visualization and plotting tools
|   ├── preparedata.py         # Data loading, preprocessing, and formatting
|   └── wshrRelabelLight.py    # Data relabeling and helper utilities
└── src/
    ├── main.py                # Main executable script for training/evaluation
    ├── MA_models.py           # Mechanism-Data-Integration architecture implementation
    ├── forecast.py            # First-phase data-driven forecasting implementations
    ├── mechanism.py           # Second-phase mechanism-aiding modules
    ├── cnn.py                 # Convolutional Neural Network implementation
    ├── LSTM.py                # Long Short-Term Memory network implementation
    ├── convTrans.py           # Convolutional Transformer model implementation
    ├── NP_ODE.py              # Neural Process Ordinary Differential Equations implementation
    └── selfattention.py       # Self-attention modules implementation
```

## Installation

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/LaMiracle/MA-TwinTran.git
   cd <repository_directory>
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   conda create -n ts_forecast python=3.9
   conda activate ts_forecast
   ```

3. Install the required dependencies. Ensure you have the correct version of PyTorch installed for your hardware (e.g., CUDA support):
   ```bash
   pip install torch pandas numpy matplotlib
   ```

## Usage

To train or evaluate the models, you can run the primary script from the terminal. 

```bash
python src/main.py
```

*Note: You may need to specify arguments parameters inside `main.py` depending on your exact configuration for the dataset.*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please contact [song-j23@mails.tsinghua.edu.cn].
