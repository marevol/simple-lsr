# Simple LSR (Work in Progress)

This project implements a simple LSR model using a Multi-Layer Perceptron (MLP) to transform sparse TF-IDF representations of queries into enhanced representations. It supports training, sparse vectorization, and retrieval tasks with multilingual support.

## Features

- **Training**: Train the MLP model with a contrastive loss to learn query expansion and matching.
- **Sparse Vectorization**: Convert text inputs into sparse representations (word-value pairs) using TF-IDF.
- **Query Expansion**: Enhance query representations using a trained MLP model.
- **Multilingual Support**: Supports multilingual text through flexible tokenization and vectorization.

## Installation

### Requirements

- Python 3.10+
- Poetry
- PyTorch
- Transformers (Hugging Face)
- Pandas
- Scikit-learn

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/marevol/simple-lsr.git
   cd simple-lsr
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

   This will create a virtual environment and install all the necessary dependencies listed in `pyproject.toml`.

3. Activate the virtual environment created by Poetry:
   ```bash
   poetry shell
   ```

## Data Preparation

This project uses the **Amazon ESCI dataset** for training the model. You need to download the dataset and place it in the correct directory.

1. Download the dataset:
   - Download the **shopping_queries_dataset_products.parquet** and **shopping_queries_dataset_examples.parquet** files from the [Amazon ESCI dataset](https://github.com/amazon-science/esci-data).

2. Place the downloaded files in the `downloads` directory within your project folder:
   ```bash
   ./downloads/shopping_queries_dataset_products.parquet
   ./downloads/shopping_queries_dataset_examples.parquet
   ```

3. The `main.py` script is set to load the dataset from the `downloads` directory by default. If you wish to place the files elsewhere, modify the paths in the script accordingly.

## Usage

### Running the Sample Script

The `main.py` script demonstrates how to use the **Amazon ESCI dataset** to train the LSR model, save it, and then use the trained model to convert text into sparse vectors for retrieval.

To run the sample execution with the Amazon ESCI dataset:

```bash
poetry run python main.py
```

This script performs the following steps:

1. **Training**: It loads the product titles and queries from the Amazon ESCI dataset, trains the MLP model on the query-product pairs, and saves the trained model.
2. **Sparse Vectorization**: After training, the model is used to convert a sample query into an expanded sparse vector representation.
3. **Query Expansion**: It demonstrates query expansion by transforming the sparse query vectors using the trained MLP model.

You can modify the script or dataset paths as needed.

### File Structure

- `main.py`: The main entry point for running the sample execution using the Amazon ESCI dataset.
- `simple_lsr/vectorization.py`: Contains the `SparseVectorizer` class for converting text into sparse TF-IDF vectors.
- `simple_lsr/model.py`: Defines the `SimpleLSR` model architecture.
- `simple_lsr/train.py`: Handles the training process for the LSR model with contrastive loss.
- `simple_lsr/evaluate.py`: Contains functions for evaluating the model using positive and negative pairs.

### Output

Once the script completes, the following will happen:

1. A trained model will be saved in the `lsr_model` directory.
2. Expanded sparse vector representations for the example queries will be printed in the console.
3. The transformed vectors will show the enhanced query representations.

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for more details.
