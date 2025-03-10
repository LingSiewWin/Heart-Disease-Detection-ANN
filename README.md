1.Install dependencies:
pip install -r requirements.txt

2.Verify installation:
python -c "import torch; print(torch.__version__)"

3.Prepare dataset:
Dataset--Heart-Disease-Prediction-using-ANN.csv

4.Train the model.

Model Architecture
The ANN consists of:

Input layer (13 features)
2 hidden layers (64 and 32 neurons)
Batch normalization
Dropout (30%)
Sigmoid output layer