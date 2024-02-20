import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth

class SpotifyClient:
    def __init__(self, auth_manager=None):
        self.sp = spotipy.Spotify(auth_manager=auth_manager)

    @staticmethod
    def get_auth_manager(scope, redirect_uri=None):
        client_id = os.getenv('SPOTIPY_CLIENT_ID')
        client_secret = os.getenv('SPOTIPY_CLIENT_SECRET')
        redirect_uri = redirect_uri or os.getenv('SPOTIPY_REDIRECT_URI')
        
        if not all([client_id, client_secret, redirect_uri]):
            raise ValueError("Spotify credentials or redirect URI are missing from environment variables")

        return SpotifyOAuth(scope=scope,
                            redirect_uri=redirect_uri,
                            client_id=client_id,
                            client_secret=client_secret,
                            show_dialog=True)


    def fetch_top_tracks(self):
        results = self.sp.current_user_top_tracks()
        return results
