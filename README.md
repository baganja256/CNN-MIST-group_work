# Group 1 CNN on MNIST

Course practical exploring convolutional neural networks on the MNIST handwritten-digit dataset. The project is available as both a Python script and a Jupyter notebook.

## Setup

Create and activate a virtual environment, then install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Run

Run the script:

```bash
python group1_cnn_mnist.py
```

Or open `group1_cnn_mnist.ipynb` in Jupyter and run its cells. The MNIST dataset is loaded through Keras and may be downloaded the first time it is used. The script saves plots in the current directory; these generated image files are excluded from version control.
