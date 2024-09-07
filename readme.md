# Datumaro-GUI

Datumaro-GUI is a graphical user interface (GUI) built with Streamlit to make working with computer vision datasets using Datumaro easier and more intuitive. This project aims to provide an easy-to-use interface for registering and merging annotations.

## Features

- **Register new datasets**: Upload and process images and annotations to create new datasets.
- **Merge datasets**: Combine existing datasets with new data and split them into training and validation sets.
- **AWS S3 Integration**: Load existing datasets from S3 and upload the processed datasets back to S3.

## Project Structure

```
datumaro-gui/
├── app.py
├── utils.py
├── pages/
│   ├── merge.py
│   ├── new.py
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
- **credentials/aws.yaml**: Stores the AWS credentials for interacting with S3.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ccomkhj/datumaro-gui.git
    cd datumaro-gui/
    ```

2. Create a virtual environment and activate it:
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


## Contributing

Contributions are welcome! Feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Datumaro](https://github.com/cvat-ai/datumaro) for providing a powerful dataset management tool.
- [Streamlit](https://www.streamlit.io/) for providing a platform to build interactive GUI applications.
