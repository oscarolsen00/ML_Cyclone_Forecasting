# ML Cyclone Forecasting

This repository contains code from my master's thesis, focused on using machine learning techniques to forecast **cyclogenesis** (the formation of tropical or extratropical cyclones). The project combines environmental data processing, deep learning architectures (ResNet, SENet), and time-series analysis.

---

## Project Structure

- `import_dataset.py` — Handles loading and preprocessing of raw meteorological data.
- `Data_processing.py` — Contains functions for normalizing, cleaning, and transforming data for modeling.
- `Resnet_model.py` — Implements a ResNet-based architecture for classification or regression tasks.
- `SENet.py` — Implements a Squeeze-and-Excitation block 
- `LICENSE` — Distributed under the GNU General Public License v3.
- `README.md` — You're reading it!

---

## Goal

The primary goal is to evaluate whether deep learning models, particularly convolutional neural networks with residual and attention mechanisms, can improve the prediction of cyclone formation using historical and geophysical data.

---

## Requirements

- Python 3.8+
- `pandas`
- `numpy`
- `torch` (PyTorch)
- `scikit-learn`
- `matplotlib` or `seaborn` (optional for visualizations)

> You can install dependencies via:
```bash
pip install -r requirements.txt
```


## How to Run

1. Clone the repository:
   ```bash
   git clone git@github.com:oscarolsen00/ML_Cyclone_Forecasting.git
   cd ML_Cyclone_Forecasting
   ```

2. Prepare your data using:
   ```bash
   python import_dataset.py
   ```

3. Process the data:
   ```bash
   python Data_processing.py
   ```

4. Train the model (example for ResNet):
   ```bash
   python Resnet_model.py
   ```

---

## License

This project is licensed under the **GNU General Public License v3.0** – see the [LICENSE](./LICENSE) file for details.

---

## Contact

For questions or collaborations, feel free to reach out to:  
**Oscar Olsen**  
oscar@olsenplanet.com 

