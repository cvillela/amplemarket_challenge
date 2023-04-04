# Amplemarket's Machine Learning Engineer Challenge
### Caio Villela

## Model Training Notebook

This repository contains a Jupyter notebook for training two LGBM models to classify between B2B and B2C companies using the dataset in data.csv. The notebook also saves scalers and models used for the inference API and contains detailed explanations.

The notebook was trained using a Google Colab PRO environment with GPU acceleration.
<br>
<br>
## Flask API
The repository also includes a Flask API `app.py` that runs a localhost server at `http://localhost:5000/` and offers a /predict endpoint for company B2C/B2B prediction. The expected format of the POST request is:

```python
{
    'name': str , 'alexa_rank': int, 'city': str, 'state': str, 'country': str, 'hq': str, 'website': str, 'employees_on_linkedin':int, 
    'followers': int, 'founded': int, 'industry': str, 'linkedin_url': str, 'overview': str, 'ownership_type': str, 'sic_codes': int, 
    'size': str, 'specialties': str,'total_funding': int, 'technologies': str, 'company_hubs': str, 'events': str, 'categories': str
}
````
The output is of the format `{'prediction_b2c': int, 'prediction_b2b': int}`, where int can be 1 or 0 for True or False predictions on each category.
<br>
<br>
### Installation
To install the necessary packages, run:
```
pip install -r requirements.txt
```

It is recommended to use a virtual environment.

### BART Model
For zero-shot classification and feature extraction from text contained in the POST request's body, we used Facebook's BART model. To use a downloaded BART model, it should be under a `bart-classifier` directory in the project directory. Run `python app.py --load_pretrained_bart True` to use the downloaded model. If the `--load_pretrained_bart` flag is not specified, the model will be downloaded at runtime.

Note that the model is large, and if the code were in production, it should be hosted on a different server. The model can be manually downloaded at https://huggingface.co/facebook/bart-large-mnli/tree/main, and the bart-classifier directory must contain the following files:

```
 config.json
 merges.txt
 pytorch_model.bin
 special_tokens_map.json
 tokenizer.json
 tokenizer_config.json
 vocab.json.
```
We recommend using GPU-ready hardware for inference.

### Testing
The repository also includes a script for testing POST requests to the Flask app at `http://localhost:5000/predict`. The script takes dataset rows at random and outputs the actual value for the company categories and the predicted values.
