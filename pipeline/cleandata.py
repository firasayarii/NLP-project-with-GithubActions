def datacleaning(df) :
    import re
    import spacy
    def clean_text(text):
    # Lowercasing
        text = text.lower()

        # Remove special characters and numbers
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    df['cleaned_review'] = df['review'].apply(clean_text)
        # Charger le modèle de langue anglais
    nlp = spacy.load("en_core_web_sm")

    # Récupérer la liste des stop words
    stop_words_spacy = nlp.Defaults.stop_words

    def remove_stopwords(text):
        tokens = text.split() 
        filtered_tokens = [word for word in tokens if word not in stop_words_spacy]  
        return ' '.join(filtered_tokens) 
    df['cleaned_review'] = df['cleaned_review'].apply(remove_stopwords)

    return df