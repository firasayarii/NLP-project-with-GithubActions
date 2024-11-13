# Initialize the Porter Stemmer
def stemming(df):
    import pandas as pd
    from nltk.stem import PorterStemmer
    from tqdm import tqdm  
    stemmer = PorterStemmer()

    def stem_tokens(tokens):
        return [stemmer.stem(token) for token in tokens]

    # Apply stemming to the tokens
    df['stemmed_tokens'] = df['tokens'].progress_apply(stem_tokens)

    # Display the results
    return df