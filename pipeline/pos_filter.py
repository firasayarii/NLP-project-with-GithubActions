
def POS_filter(df):
    import pandas as pd
    import spacy

# Load the SpaCy language model
    nlp = spacy.load('en_core_web_sm')
    def filter_text(sequence):
        s=[]
        for token in sequence :
            s.append(str(token))
        return ' '.join(s)
    def pos_tagging_and_filter(tokens):
        # Apply POS tagging
        doc=nlp(' '.join(tokens))

        # Filter only nouns and verbs
        filtered_tokens = [token for token in doc if token.pos_ == 'NN' or token.pos_ == 'ADJ' or token.pos_ == 'VERB']

        return filtered_tokens

    # Apply POS tagging and filtering to lemmatized tokens
    df['filtered_tokens'] = df['lemmatized_tokens'].apply(pos_tagging_and_filter)

    df['filtered_text'] = df['filtered_tokens'].apply(filter_text)


    return df