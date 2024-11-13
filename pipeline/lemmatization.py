
def lemma(df):
    import pandas as pd
    import spacy

# Load the SpaCy language model
    nlp = spacy.load('en_core_web_sm')
    def lemmatize_tokens(tokens):
        doc = nlp(" ".join(tokens))  
        return [token.lemma_ for token in doc]

    # Apply lemmatization to the filtered tokens
    df['lemmatized_tokens'] = df['tokens'].apply(lemmatize_tokens)

    return df