ASTROCORTEX — FINAL SIMULATION PACKAGE
======================================

This package contains the Streamlit dashboard and the training script required
to generate its synthetic dataset and saved machine-learning models.

QUICK START (Windows)
---------------------
1. Extract this ZIP file.
2. Open the extracted AstroCortex_Final folder.
3. Click the folder address bar, type cmd, and press Enter.
4. Run:

   python -m pip install -r requirements.txt

5. Generate the dataset and train all three models:

   python train_model.py

6. Start the dashboard:

   python -m streamlit run app.py

7. Open the local URL shown in the terminal (usually http://localhost:8501).

WHAT IS INCLUDED
----------------
- app.py: four-tab interactive Streamlit dashboard.
- train_model.py: creates a reproducible 2,400-row synthetic dataset, trains
  Logistic Regression, Decision Tree and Random Forest, and saves metrics.
- requirements.txt: Python package dependencies.
- data/: generated CSV files will be saved here.
- models/: trained model files will be saved here.

SCIENTIFIC INTEGRITY
--------------------
All training examples and sensor trends are synthetic. Risk labels are created
using a transparent illustrative scoring rule for demonstration. They are not
real lunar measurements, NASA data, experimental findings, or a validated
physical model of lunar dust. Accuracy and other metrics describe performance
against this synthetic dataset only. The dashboard is a research prototype,
not flight-qualified hardware or an operational safety system.

TROUBLESHOOTING
---------------
- If packages are missing, rerun: python -m pip install -r requirements.txt
- If model files or data are missing, run: python train_model.py
- If Streamlit is already running, stop it with Ctrl+C and restart it.
