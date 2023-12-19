import collections
import numpy as np
import pandas as pd
import pretty_midi
import os
import json
import tables
import nltk

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')

DATA_PATH = 'data'
RESULTS_PATH = ''
SCORE_FILE = os.path.join(RESULTS_PATH, 'match_scores.json')

results = []
author_length_results = []

# Utility functions for retrieving paths

# Given an MSD ID, generate the path prefix.
# E.g. TRABCD12345678 -> A/B/C/TRABCD12345678
def msd_id_to_dirs(msd_id):
    return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)

# Given an MSD ID, return the path to the corresponding mp3
def msd_id_to_mp3(msd_id):
    return os.path.join(DATA_PATH, 'msd', 'mp3',
                        msd_id_to_dirs(msd_id) + '.mp3')

# Given an MSD ID, return the path to the corresponding h5
def msd_id_to_h5(msd_id):
    return os.path.join(RESULTS_PATH, 'lmd_matched_h5', msd_id_to_dirs(msd_id) + '.h5')

# Given an MSD ID and MIDI MD5, return path to a MIDI file.
# kind should be one of 'matched' or 'aligned'.
def get_midi_path(msd_id, midi_md5, kind):
    return os.path.join(RESULTS_PATH, 'lmd_{}'.format(kind),
                        msd_id_to_dirs(msd_id), midi_md5 + '.mid')

#This function was taken from https://github.com/SirawitC/NLP-based-music-processing-for-composer-classification
def midi_to_notes(midi_file: str) -> pd.DataFrame:
  pm = pretty_midi.PrettyMIDI(midi_file)
  instrument = pm.instruments[0]
  notes = collections.defaultdict(list)

  sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
  prev_start = sorted_notes[0].start

  for note in sorted_notes:
    start = note.start
    end = note.end
    notes['pitch'].append(note.pitch)
    notes['start'].append(start)
    notes['end'].append(end)
    notes['step'].append(start - prev_start)
    notes['duration'].append(end - start)
    notes['velocity'].append(note.velocity)
    prev_start = start
  return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

def main():
    with open(SCORE_FILE) as f:
        scores = json.load(f)
    # Grab a Million Song Dataset ID from the scores dictionary
    msd_id = list(scores.keys())[1234]
    print('Million Song Dataset ID {} has {} MIDI file matches:'.format(
        msd_id, len(scores[msd_id])))
    for midi_md5, score in scores[msd_id].items():
        print('  {} with confidence score {}'.format(midi_md5, score))

    while True:
        # Grab an MSD ID and its dictionary of matches
        msd_id, matches = scores.popitem()
        # Grab a MIDI from the matches
        midi_md5, score = matches.popitem()
        # Construct the path to the aligned MIDI
        aligned_midi_path = get_midi_path(msd_id, midi_md5, 'aligned')

        pm = pretty_midi.PrettyMIDI(aligned_midi_path)
        # Look for a MIDI file which has lyric and key signature change events
        if len(pm.lyrics) > 5 and len(pm.key_signature_changes) > 0:
            break

    with tables.open_file(msd_id_to_h5(msd_id)) as h5:
        print('ID: {}'.format(msd_id))
        print('"{}" by {} on "{}"'.format(
            h5.root.metadata.songs.cols.title[0],
            h5.root.metadata.songs.cols.artist_name[0],
            h5.root.metadata.songs.cols.release[0]))
        print('Top 5 artist terms:', ', '.join(
            [str(l) for l in list(h5.root.metadata.artist_terms)[:5]]))


    authors = {}

    midi_lyrics_author = []
    number_iterations = 0

    for msd_id, midi_matches in scores.items():
        if number_iterations >= 750:
            break  
        h5_file = msd_id_to_h5(msd_id)
        with tables.open_file(h5_file) as h5:
            author = h5.root.metadata.songs.cols.artist_name[0].decode('utf-8')
            if author in authors:
                authors[author] += len(matches)
            else:
                authors[author] = len(matches)
    
        for midi_md5 in midi_matches.keys(): 
            midi_path = get_midi_path(msd_id, midi_md5, 'aligned')
            try:
                pm = pretty_midi.PrettyMIDI(midi_path)
                if len(pm.lyrics) > 5 and len(pm.key_signature_changes) > 0:
                    midi_lyrics_author.append((midi_path, author))
            except Exception as e:
                print(f"Error processing {midi_path}: {e}")

    
        number_iterations += 1  # Increment counter after processing each MSD ID

    sorted_artists = sorted(
    authors.items(), key=lambda x: x[1])
    print(sorted_artists)


   # Define a fixed length for feature vectors
    FIXED_LENGTH = 1000  

    X_midi, X_lyrics, y, X_lyrics_no_stop_words = [], [], [], []

    midi_case = True
    lyrics_case = True
    combined_case = True

    stop_words = set(stopwords.words('english'))

    for midi_path, author in midi_lyrics_author:
        if authors[author] > 4:
            try:
                pm = pretty_midi.PrettyMIDI(midi_path)
                if pm.lyrics:  # Process only if lyrics are present
                    if lyrics_case:
                        lyrics_no_stop_words = []
                        lyrics = ' '.join([lyric.text for lyric in pm.lyrics])
                        for lyric in pm.lyrics:
                            words = word_tokenize(lyric.text)
                            filtered_lyrics = [word for word in words if word.lower() not in stop_words]
                            lyrics_no_stop_words = ' '.join(filtered_lyrics)
                        X_lyrics.append(lyrics)
                        X_lyrics_no_stop_words.append(lyrics_no_stop_words)


                    if midi_case:
                        midi_features = midi_to_notes(midi_path)
                        features_flattened = midi_features.values.flatten()
                
                        # Pad or truncate the feature vector
                        if len(features_flattened) > FIXED_LENGTH:
                            features_flattened = features_flattened[:FIXED_LENGTH]
                        else:
                            features_flattened = np.pad(features_flattened, (0, FIXED_LENGTH - len(features_flattened)))

                        X_midi.append(features_flattened)
                    
                    y.append(author)
            except Exception as e:
                print(f"Error processing {midi_path}: {e}")

    # Vectorize lyrics
    if lyrics_case:
        vectorizer = CountVectorizer()
        print(X_lyrics)
        X_lyrics_vectorized = vectorizer.fit_transform(X_lyrics).toarray()
        print("\nLyrics Only")
        train_and_test(np.array(X_lyrics_vectorized), np.array(y), 'Lyric', 'Yes')

        print("\nLyrics Only, No stop words")
        X_lyrics_vectorized_no_stop_words = vectorizer.fit_transform(X_lyrics_no_stop_words).toarray()
        train_and_test(np.array(X_lyrics_vectorized_no_stop_words), np.array(y), 'Lyric', 'No')

    if midi_case:
        print("MIDI Only")
        train_and_test(np.array(X_midi), np.array(y), 'MIDI', 'NA')

    if combined_case:
        # Combine MIDI and lyrics features
        X_combined = np.concatenate((np.array(X_midi), X_lyrics_vectorized), axis=1)
        print("\nCombined with stop words")
        train_and_test(X_combined, np.array(y), 'Lyric+MIDI', 'Yes')

        X_combined_no_stop_words = np.concatenate((np.array(X_midi), X_lyrics_vectorized_no_stop_words), axis=1)
        print("\nCombined without stop words")
        train_and_test(X_combined_no_stop_words, np.array(y), 'Lyric+MIDI', 'No')

def train_and_test(X, y, configuration, stop_words):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    model_trainers = [
    (train_logistic_regression, "Logistic Regression"),
    (train_linear_svm, "Linear SVM")
    ]

    # Iterate over each model, train, predict, and print accuracy
    for train_func, model_name in model_trainers:

        model = train_func(X_train_scaled, y_train, X_val_scaled, y_val)

        y_pred = model.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)

        print(f"{model_name} Accuracy: {accuracy:.2f}")

        results.append((model_name, configuration, stop_words, accuracy))

    
    #scaling introduces negative values which can be an issue for NB model
    NB_model = train_naive_bayes(X_train, y_train, X_val, y_val)

    y_pred = NB_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Naive Bayes Accuracy: {accuracy:.2f}")

    results.append(('Naive Bayes', configuration, stop_words, accuracy))

def train_logistic_regression(X_train_vec, y_train, X_val_vec, y_val):
    best_val_accuracy = 0
    best_hyperparameters = {}

    fit_intercept_options = [True, False]
    penalty_options = ['l1', 'l2']

    for fit_intercept in fit_intercept_options:
        for penalty in penalty_options:
            model = LogisticRegression(
                fit_intercept=fit_intercept, 
                penalty=penalty, 
                solver='liblinear',
                max_iter=1000
            )
            model.fit(X_train_vec, y_train)

            y_val_pred = model.predict(X_val_vec)
            val_accuracy = accuracy_score(y_val, y_val_pred)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_hyperparameters = {
                    'fit_intercept': fit_intercept,
                    'penalty': penalty,
                    'solver':'liblinear'
                }

    final_model = LogisticRegression(**best_hyperparameters)
    final_model.fit(X_train_vec, y_train)

    return final_model

def train_naive_bayes(X_train_vec, y_train, X_val_vec, y_val):
    best_val_accuracy = 0
    alpha_values = [0.1, 0.5, 1.0, 2.0]  
    best_hyperparameter = 0.1 #this will get updated if there is a better one

    for alpha_value in alpha_values:
                        model = MultinomialNB(alpha=alpha_value)
                        model.fit(X_train_vec, y_train)

                        y_val_pred = model.predict(X_val_vec)
                        val_accuracy = accuracy_score(y_val, y_val_pred)

                        if val_accuracy > best_val_accuracy:
                            best_val_accuracy = val_accuracy
                            best_hyperparameter = alpha_value
    

    final_model = MultinomialNB(alpha=best_hyperparameter)
    final_model.fit(X_train_vec, y_train)
    return final_model

def train_linear_svm(X_train_vec, y_train, X_val_vec, y_val):
    best_val_accuracy = 0
    best_hyperparameters = {}

    fit_intercept_options = [True, False]

    penalty_options = ['l1', 'l2']

    for fit_intercept in fit_intercept_options:
            for penalty in penalty_options:
                model = LinearSVC(dual=False, fit_intercept=fit_intercept, penalty=penalty, max_iter=20000)
                model.fit(X_train_vec, y_train)

                y_val_pred = model.predict(X_val_vec)
                val_accuracy = accuracy_score(y_val, y_val_pred)

                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_hyperparameters = {
                        'max_iter': 20000,
                        'dual': False,
                        'penalty': penalty,
                        'fit_intercept': fit_intercept
                    }

    # Train final model with best hyperparameters for Linear SVM
    final_model = LinearSVC(**best_hyperparameters)
    final_model.fit(X_train_vec, y_train)
    return final_model


if __name__ == "__main__":
    main()
    df = pd.DataFrame(results, columns=['Model', 'Configuration', 'Stop Words', 'Accuracy'])
    df = df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)

    print(df)

    # Filter out 'No Stop Words' cases
    filtered_results = [row for row in results if row[2] != 'No']
    df1 = pd.DataFrame(filtered_results, columns=['Model', 'Configuration', 'Stop Words', 'Accuracy'])
    df1 = df1.drop('Stop Words', axis=1).sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
    print(df1)

    # Filter for only 'Lyric' and 'Lyric + MIDI' cases
    filtered_results_2 = [row for row in results if row[1] in ['Lyric', 'Lyric+MIDI']]
    df2 = pd.DataFrame(filtered_results_2, columns=['Model', 'Configuration', 'Stop Words', 'Accuracy'])
    df2 = df2.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
    print(df2)