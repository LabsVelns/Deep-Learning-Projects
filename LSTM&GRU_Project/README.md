# Next Word Predictor (LSTM)

Project that trains an LSTM-based language model on Shakespeare's "Hamlet" to predict the next word given a phrase, and exposes a small Streamlit web UI for interactive prediction.

## Contents
- `train.ipynb` - Notebook to preprocess data, build/train the LSTM model, save the trained model and tokenizer.
- `app.py` - Streamlit app that loads the saved model + tokenizer and predicts the next word for user input.
- `hamlet.txt` - Raw training text (Shakespeare - Hamlet).
- `requirements.txt` - Python dependencies.
- `next_word_predictor_lstm.h5` - Recommended location/name for the trained Keras model (created by the notebook).
- `tokenizer.pickle` (or `Tokenizer.pickle`) - Saved tokenizer (pickle) used at inference.

## Setup

1. Clone or open this project folder.
2. Create a Python virtual environment and activate it (Windows example):
   - python -m venv .venv
   - .venv\Scripts\activate
3. Install dependencies:
   - pip install -r requirements.txt

Notes:
- Use Python 3.8+ and TensorFlow 2.x (works with tf.keras).
- On Windows filenames are case-insensitive; ensure the tokenizer filename used by `app.py` matches the saved file name.

## Training

Open and run `train.ipynb` in Jupyter / VS Code notebook:

- The notebook downloads/loads the Hamlet text, tokenizes and creates n-gram sequences, pads sequences, and builds an LSTM model.
- Train the model (the notebook uses EarlyStopping).
- After training the notebook saves:
  - model: `next_word_predictor_lstm.h5`
  - tokenizer: `Tokenizer.pickle` (or `tokenizer.pickle`)

If you re-run training, ensure the saved model/tokenizer names match those expected by `app.py`.

## Running the Streamlit App (Inference)

From the project folder (Windows example):

- cd "d:\Projects\Lstm Projects"
- streamlit run app.py

App usage:
- Enter a phrase in the input field and click "Predict".
- The app uses the saved tokenizer and model to predict the most likely next word.

## Implementation Details

- Tokenization: Keras Tokenizer is used to build a vocabulary and encode text.
- Input sequences: n-gram sequences are created from lines of the Hamlet text.
- Model: Embedding -> LSTM -> Dropout -> LSTM -> Dense(softmax).
- Inference: `app.py` pads input sequences to the model's expected length and returns the predicted word index mapped back to the word string.

## Common Issues & Tips

- FileNotFoundError for `next_word_predictor_lstm.h5` or tokenizer: ensure these files exist in the project root and names match `app.py`.
- If training is slow, use a GPU-enabled environment or reduce batch size/model size.
- Version issues: ensure TensorFlow/Keras versions are compatible (TensorFlow 2.x recommended).
- If predictions are poor, consider:
  - using more text or different corpus,
  - tuning model architecture / hyperparameters,
  - increasing embedding or LSTM sizes,
  - using more training epochs with appropriate EarlyStopping.

## Extending
- Replace `hamlet.txt` with other corpora to specialize the predictor.
- Add support for top-k predictions instead of only top-1.
- Package the model as an API using FastAPI or Flask for programmatic access.

## License & Acknowledgements
- MIT-style license (adapt as needed).
- Dataset: public-domain Shakespeare (Project Gutenberg).
