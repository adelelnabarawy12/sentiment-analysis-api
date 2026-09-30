from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import namedtuple
import joblib

def process_text(text: str, english_stops: list[str] = set(stopwords.words('english')), joinded = True):
    tokens = word_tokenize(str(text).strip().lower())
    stemmer = PorterStemmer()
    out_tokens = []
    for token in tokens:
        if token.isalnum() and token not in english_stops:
            out_tokens.append(stemmer.stem(token))
        # End If
    # End For
    return " ".join(out_tokens) if joinded else out_tokens
# End Func

def inference(text, model, vectorizer, labels_decoder):
    features = vectorizer.transform([process_text(_) for _ in text])
    _confidence_scores = model.predict_proba(features)
    confidence_scores, labels = _confidence_scores.max(axis = 1), _confidence_scores.argmax(axis = 1)
    out = [{"label": labels_decoder[lbl].capitalize(), "confidence_score": float(cf)} for cf, lbl in zip(confidence_scores, labels)]
    return out
# End Func

def load_model(vectorizer_path, model_path):
    SKModel = namedtuple("SKModel", ["vectorizer", "model"])
    return SKModel(
        vectorizer = joblib.load(vectorizer_path),
        model = joblib.load(model_path)
    )
# End Func