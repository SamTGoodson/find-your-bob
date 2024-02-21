import pandas as pd
import numpy as np
import os
import ast

import time

import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.exceptions import SpotifyException

from scipy.spatial import distance

from sklearn.neighbors import NearestNeighbors


scope = "user-library-read user-top-read playlist-modify-public"
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))

manual_catagorical_cols = ['mode', 'key', 'time_signature']
feature_columns = ['danceability_mean', 'energy_mean', 'loudness_mean','speechiness_mean',
                    'acousticness_mean', 'instrumentalness_mean','liveness_mean', 'valence_mean',
                      'tempo_mean', 'mode_mode', 'key_mode','time_signature_mode'] 

def process_dataframe(df, manual_categorical_cols):
    
    processed_data = {}
    df = df.drop(columns=['id'])

    for col in df.columns:
        if col in manual_categorical_cols:
            processed_data[col] = df[col].mode()[0]
        else:
            processed_data[col] = df[col].mean()
    
    return processed_data

def process_dataframe_with_variance_weighting(df, manual_categorical_cols):
    processed_data = {}
    
    df = df.drop(columns=['id'])
    
    variances = df.var()
    
    variance_weights = variances / variances.sum()
    
    for col in df.columns:
        if col in manual_categorical_cols:
            processed_data[col] = df[col].mode()[0]
        else:
            weighted_column = df[col] * variance_weights[col]
            processed_data[col] = weighted_column.mean()
    
    return processed_data

def find_closest_album(user_raw,album_features, feature_columns):
    max_similarity = -1  
    closest_album = None

    averages_features = [user_raw[col] for col in feature_columns]
    user_features = np.array(averages_features).flatten()

    for index, row in album_features.iterrows():
        album_features = row[feature_columns].tolist()  
        similarity = 1 - distance.cosine(user_features, album_features)  
        if similarity > max_similarity:
            max_similarity = similarity
            closest_album = row['album_name']
            images = ast.literal_eval(row['images']) 
            if images:  
                image_url = images[0]['url'] 

    return closest_album,image_url


def get_recommended_songs(songs,user_info):
    print(songs.head(), user_info)
    X = songs.drop(columns=['id', 'track_name', 'album_name'])
    feature_columns = X.columns.tolist()
    print(feature_columns)
    knn = NearestNeighbors(n_neighbors=30)
    knn.fit(X)
    user_profile = np.array([user_info[feature] for feature in feature_columns]).reshape(1, -1)
    print(user_profile)
    distances, indices = knn.kneighbors(user_profile)
    print(distances, indices)
    recommended_songs_df = songs.iloc[indices[0]][['track_name', 'id']]
    return recommended_songs_df

def create_and_fill_playlist(recommended_songs_df, user):
    print('starting playlist creation')
    try:
        user_id = user['id']
        print("Creating playlist for user:", user_id)
        playlist = sp.user_playlist_create(user_id, "FindYourBob", public=True, collaborative=False, description='Discovering your personal slice of Bob')
        print("Playlist created. ID:", playlist['id'])

        track_ids = recommended_songs_df['id'].tolist()
        if track_ids:  
            sp.user_playlist_add_tracks(user_id, playlist['id'], track_ids, position=None)
            print(f"{len(track_ids)} tracks added to the playlist.")
        else:
            print("No tracks to add to the playlist.")
    except Exception as e:
        print("An error occurred:", e)


def process_bob(bob_df, cat_cols, con_cols):

    processed_data = pd.DataFrame()
    
    grouped = bob_df.groupby('album_name')
    
    for col in con_cols:
        if col in bob_df.columns:
            processed_data[col + '_mean'] = grouped[col].mean()
    
    for col in cat_cols:
        if col in bob_df.columns:
            processed_data[col + '_mode'] = grouped[col].apply(lambda x: x.mode()[0] if not x.mode().empty else None)
    
    processed_data = processed_data.reset_index()
    
    return processed_data
