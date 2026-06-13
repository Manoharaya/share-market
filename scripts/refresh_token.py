import os
import sys
import webbrowser
import urllib.parse
import requests
from http.server import HTTPServer, BaseHTTPRequestHandler
from dotenv import load_dotenv, set_key

# Path to the .env file in the root directory
ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))

def load_credentials():
    load_dotenv(ENV_PATH)
    api_key = os.getenv("UPSTOX_API_KEY")
    api_secret = os.getenv("UPSTOX_API_SECRET")
    return api_key, api_secret

def update_env_token(access_token):
    # Update the .env file with the generated access token
    set_key(ENV_PATH, "UPSTOX_ACCESS_TOKEN", access_token)
    print("\n[SUCCESS] UPSTOX_ACCESS_TOKEN successfully updated in .env!")

class OAuthHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress logging server requests to keep console clean
        return

    def do_GET(self):
        # Parse the query parameters
        parsed_url = urllib.parse.urlparse(self.path)
        query_params = urllib.parse.parse_qs(parsed_url.query)
        
        auth_code = query_params.get("code")
        
        if auth_code:
            auth_code = auth_code[0]
            print(f"[INFO] Received Authorization Code: {auth_code}")
            
            try:
                # Exchange auth code for access token via Upstox API
                api_key, api_secret = load_credentials()
                redirect_uri = "http://localhost:8000"
                
                token_url = "https://api.upstox.com/v2/login/authorization/token"
                payload = {
                    "code": auth_code,
                    "client_id": api_key,
                    "client_secret": api_secret,
                    "redirect_uri": redirect_uri,
                    "grant_type": "authorization_code"
                }
                headers = {
                    "accept": "application/json",
                    "Content-Type": "application/x-www-form-urlencoded"
                }
                
                response = requests.post(token_url, data=payload, headers=headers)
                response_data = response.json()
                
                if "access_token" in response_data:
                    access_token = response_data["access_token"]
                    # Update the .env file
                    update_env_token(access_token)
                    
                    # Show success in the browser
                    self.send_response(200)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    
                    success_html = """
                    <html>
                    <head>
                        <title>Upstox Authentication Success</title>
                        <style>
                            body {
                                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                                background: #0d1117;
                                color: #c9d1d9;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                height: 100vh;
                                margin: 0;
                            }
                            .container {
                                text-align: center;
                                padding: 30px;
                                background: #161b22;
                                border-radius: 10px;
                                border: 1px solid #30363d;
                                box-shadow: 0 4px 12px rgba(0,0,0,0.5);
                            }
                            h1 { color: #58a6ff; margin-bottom: 10px; }
                            p { font-size: 16px; }
                        </style>
                    </head>
                    <body>
                        <div class="container">
                            <h1>Upstox Authentication Successful!</h1>
                            <p>The access token has been saved to your <code>.env</code> file.</p>
                            <p>You can now safely close this tab and return to the terminal.</p>
                        </div>
                    </body>
                    </html>
                    """
                    self.wfile.write(success_html.encode("utf-8"))
                    
                    # Signal the server to stop
                    self.server.token_retrieved = True
                else:
                    error_msg = response_data.get("errors", [{}])[0].get("message", "Unknown error")
                    raise Exception(f"{error_msg} ({response_data})")
                
            except Exception as e:
                print(f"[ERROR] Failed to exchange code for session: {e}")
                self.send_response(500)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                error_html = f"<html><body><h1>Authentication Failed</h1><p>Error: {e}</p></body></html>"
                self.wfile.write(error_html.encode("utf-8"))
        else:
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h1>Bad Request</h1><p>Missing auth code</p></body></html>")

def run_local_server(api_key, api_secret):
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, OAuthHandler)
    httpd.token_retrieved = False
    
    redirect_uri = urllib.parse.quote("http://localhost:8000", safe="")
    login_url = f"https://api.upstox.com/v2/login/authorization/dialog?response_type=code&client_id={api_key}&redirect_uri={redirect_uri}"
    
    print("\n" + "="*60)
    print("UPSTOX OAUTH FLOW")
    print("="*60)
    print(f"Opening login URL in your browser:\n{login_url}\n")
    print("If it doesn't open automatically, copy and paste the URL above into your browser.")
    print("Waiting for redirection on http://localhost:8000 ...")
    print("="*60 + "\n")
    
    # Open the browser
    webbrowser.open(login_url)
    
    # Run the server until we retrieve the token
    while not httpd.token_retrieved:
        httpd.handle_request()

def main():
    api_key, api_secret = load_credentials()
    
    if not api_key or not api_secret:
        print("\n" + "!"*60)
        print("ERROR: UPSTOX_API_KEY or UPSTOX_API_SECRET is missing in .env!")
        print("Please configure them in your .env file before running this script.")
        print("Required fields:")
        print("  - UPSTOX_API_KEY")
        print("  - UPSTOX_API_SECRET")
        print("!"*60 + "\n")
        sys.exit(1)
        
    run_local_server(api_key, api_secret)

if __name__ == '__main__':
    main()
