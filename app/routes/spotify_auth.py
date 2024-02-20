from flask import Blueprint, redirect, request, session, url_for
from app.spotify_client import SpotifyClient

bp = Blueprint('spotify_auth', __name__, url_prefix='/spotify')

@bp.route('/login')
def login():
    scope = "user-library-read user-top-read playlist-modify-public"
    redirect_uri = "http://localhost:5000/spotify/callback"  # Ensure this matches your app's redirect URI
    auth_manager = SpotifyClient.get_auth_manager(scope=scope, redirect_uri=redirect_uri)
    auth_url = auth_manager.get_authorize_url()
    return redirect(auth_url)

@bp.route('/callback')
def callback():
    auth_manager = SpotifyClient.get_auth_manager(scope="user-library-read user-top-read playlist-modify-public",
                                                  redirect_uri="http://localhost:5000/spotify/callback")
    code = request.args.get('code')
    token_info = auth_manager.get_access_token(code)
    session['token_info'] = token_info
    return redirect(url_for('index'))  # Redirect to the main page or a dashboard
