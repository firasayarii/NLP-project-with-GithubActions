def tfidf(df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    import pandas as pd
    # Initialize the TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Limit to 5000 features for simplicity

    # Fit and transform the cleaned reviews
    tfidf_embedding = tfidf_vectorizer.fit(df['filtered_text'])

    return tfidf_embedding

    # Convert to DataFrame for better visualization
    #tfidf_matrix=tfidf_embedding.transform(df['filtered_text'])
    #tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    #return tfidf_matrix