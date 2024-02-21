from flask import Flask, render_template, jsonify, request, redirect, session, url_for,render_template_string
from .utils.clustering import create_and_fill_playlist, find_closest_album, get_recommended_songs, process_dataframe
from .utils.api_calls import safe_spotify_request, get_top_features
import pandas as pd
from .spotify_client import SpotifyClient
from .data.static import feature_columns,manual_catagorical_cols,album_titles
import os
from spotipy import Spotify

current_dir = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(current_dir, 'data', 'bob_with_images.csv')
bob_features_path = os.path.join(current_dir, 'data', 'bob_features.csv')


app = Flask(__name__)
app.secret_key = os.environ.get('APP_KEY')


SPOTIPY_CLIENT_ID = 'your_spotify_client_id'
SPOTIPY_CLIENT_SECRET = 'your_spotify_client_secret'
SPOTIPY_REDIRECT_URI = 'your_redirect_uri'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    auth_manager = SpotifyClient.get_auth_manager(scope = "user-library-read user-top-read playlist-modify-public")
    return redirect(auth_manager.get_authorize_url())

@app.route('/callback')
def callback():
    required_scope = "user-library-read user-top-read playlist-modify-public"  
    auth_manager = SpotifyClient.get_auth_manager(scope=required_scope)
    code = request.args.get('code')
    token_info = auth_manager.get_access_token(code, check_cache=False)
    session['token_info'] = token_info

    return redirect(url_for('index'))

@app.route('/find_closest_album', methods=['GET', 'POST'])
def get_closest_album():
    if 'token_info' not in session:
        return redirect(url_for('login'))
    
    album_features = pd.read_csv(csv_file_path)
    data = request.get_json()  
    studio_only = data.get('studioOnly', False)
    
    if studio_only:
        bob_album = album_features[album_features['album_name'].isin(album_titles)]
    else:
        bob_album = album_features
        print("Using full dataset")
    
    token_info = session['token_info']
    sp = Spotify(auth=token_info['access_token'])
    user_raw = get_top_features(sp)
    user_df = process_dataframe(user_raw,manual_catagorical_cols) 
    closest_album, album_image_url = find_closest_album(user_df, bob_album, feature_columns)

    html_content = render_template_string('''
        <h3>{{ album_name }}</h3>
        <img src="{{ album_image_url }}" alt="Album Cover" style="width:200px;height:auto;">
    ''', album_name=closest_album, album_image_url=album_image_url)

    return html_content

@app.route('/make_playlist', methods=['POST'])
def make_playlist():
    if 'token_info' not in session:
        return redirect(url_for('login'))
    
    token_info = session['token_info']
    sp = Spotify(auth=token_info['access_token'])
    user = sp.current_user()
    user_raw = get_top_features(sp)
    user_df = process_dataframe(user_raw,manual_catagorical_cols)
    print(user_df)
    bob_raw = pd.read_csv(bob_features_path)
    recommended_songs_df = get_recommended_songs(bob_raw,user_df) 
    create_and_fill_playlist(recommended_songs_df, user) 

    return f"""<p>Your playlist was successfully created, enjoy your very own slice of Bob.</p>
    <img src="https://upload.wikimedia.org/wikipedia/commons/3/37/President_Barack_Obama_presents_American_musician_Bob_Dylan_with_a_Medal_of_Freedom.jpg" style="height: 300px; width: 300px; display: block; margin-left: auto; margin-right: auto" alt="Playlist Image">"""


if __name__ == '__main__':
    app.run(debug=True)
