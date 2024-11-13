def tokenization (df):
    import pandas as pd
    import spacy

# Load the SpaCy language model
    nlp = spacy.load('en_core_web_sm')
    # Function for tokenization
    def tokenize_text(text):
        doc = nlp(text)
        return [token.text for token in doc]



    # Apply tokenization to cleaned reviews
    df['tokens'] = df['cleaned_review'].apply(tokenize_text)

    return df