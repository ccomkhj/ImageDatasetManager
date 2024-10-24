# Datumaro-GUI

Datumaro-GUI is a graphical user interface (GUI) built with Streamlit to make working with computer vision datasets using Datumaro easier and more intuitive. This project aims to provide an easy-to-use interface for registering, merging, and filtering annotations.

## Features

- **Register new datasets**: Upload and process images and annotations to create new datasets.
- **Merge datasets**: Combine existing datasets with new data and split them into training and validation sets.
- **Filter annotations**: Apply custom filters to datasets to extract specific data based on conditions.
- **AWS S3 Integration**: Load existing datasets from S3 and upload the processed datasets back to S3.

## Project Structure

```
datumaro-gui/
├── app.py
├── utils.py
├── pages/
│   ├── merge.py
│   ├── new.py
│   ├── filter.py
├── requirements.txt
├── README.md
└── credentials/
    └── aws.yaml
```

- **app.py**: The main entry point for the Streamlit application.
- **utils.py**: Utility functions for handling file uploads, AWS S3 interactions, and other helper functions.
- **pages/**: Contains the different pages for the Streamlit app.
    - **merge.py**: Page for merging datasets.
    - **new.py**: Page for registering new annotations.
    - **filter.py**: Page for applying filters to datasets.
- **credentials/aws.yaml**: Stores the AWS credentials for interacting with S3.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ccomkhj/datumaro-gui.git
    cd datumaro-gui/
    ```

2. Create a virtual environment and activate it:
    ```bash
    conda create -n datumaro-gui python=3.11 -y
    conda activate datumaro-gui
    ```
    [Note] tested with python =< 3.11
    OR
    ```bash
    python3 -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. (if you want to use S3) Set up AWS credentials in `credentials/aws.yaml`:
    ```yaml
    aws_access_key_id: <YOUR_AWS_ACCESS_KEY_ID>
    aws_secret_access_key: <YOUR_AWS_SECRET_ACCESS_KEY>
    ```

## Usage

1. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2. Open your browser and go to `http://localhost:8501` to interact with the Datumaro-GUI.


## To read new dataset
```
-- annotations
    |- instances_train.json
    |- instances_val.json
    # subsets are train and val
-- images
    |- train
    |- val

```
or
```
-- annotations
    |- instance_default.json
    # subsets are only default
-- images
    |- train
```

## Recommended workflow
1. use `new` if it's your first dataset
2. use `merge` if your new dataset needs to be merged
3. use `filter` if you want to create the dataset selectively (after `merge` or `new`)


## Contributing

Contributions are welcome! Feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.