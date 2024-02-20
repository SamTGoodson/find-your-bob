import pandas as pd
import numpy as np
import os

import time

import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.exceptions import SpotifyException



def safe_spotify_request(call, *args, **kwargs):
    max_attempts = 5
    attempt = 0
    while attempt < max_attempts:
        try:
            return call(*args, **kwargs)
        except SpotifyException as e:
            if e.http_status == 429:  
                wait_time = int(e.headers.get('Retry-After', 30))  
                print(f"Rate limit exceeded. Retrying after {wait_time} seconds.")
                time.sleep(wait_time)
                attempt += 1
                wait_time *= 2  
            else:

                raise

    raise Exception("Maximum retry attempts reached.")


def get_top_features(sp):
    top = sp.current_user_top_tracks()
    top_df = pd.DataFrame(top['items'])
    
    features_list = []
    
    for id in top_df['id']:
        features = safe_spotify_request(sp.audio_features, id)
        
        if features[0]: 
            features_df = pd.DataFrame(features)
            features_list.append(features_df)
    

    features_df = pd.concat(features_list, ignore_index=True)
    features_df = features_df.drop(columns=['type', 'uri', 'track_href', 'analysis_url', 'duration_ms'])

    return features_df
