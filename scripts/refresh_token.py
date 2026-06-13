import os
import sys
import webbrowser
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from dotenv import load_dotenv, set_key
from kiteconnect import KiteConnect

# Path to the .env file in the root directory
ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))

def load_credentials():
    load_dotenv(ENV_PATH)
    api_key = os.getenv("ZERODHA_API_KEY")
    api_secret = os.getenv("ZERODHA_API_SECRET")
    return api_key, api_secret

def update_env_token(access_token):
    # Update the .env file with the generated access token
    set_key(ENV_PATH, "ZERODHA_ACCESS_TOKEN", access_token)
    print("\n[SUCCESS] ZERODHA_ACCESS_TOKEN successfully updated in .env!")

class OAuthHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress logging server requests to keep console clean
        return

    def do_GET(self):
        # Parse the query parameters
        parsed_url = urllib.parse.urlparse(self.path)
        query_params = urllib.parse.parse_qs(parsed_url.query)
        
        request_token = query_params.get("request_token")
        
        if request_token:
            request_token = request_token[0]
            print(f"[INFO] Received request token: {request_token}")
            
            try:
                # Exchange request token for access token
                api_key, api_secret = load_credentials()
                kite = KiteConnect(api_key=api_key)
                session = kite.generate_session(request_token, api_secret=api_secret)
                access_token = session["access_token"]
                
                # Update the .env file
                update_env_token(access_token)
                
                # Show success in the browser
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                
                success_html = """
                <html>
                <head>
                    <title>Authentication Success</title>
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
                        .token { font-family: monospace; background: #21262d; padding: 10px; border-radius: 5px; color: #8b949e; }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>Zerodha Authentication Successful!</h1>
                        <p>The access token has been saved to your <code>.env</code> file.</p>
                        <p>You can now safely close this tab and return to the terminal.</p>
                    </div>
                </body>
                </html>
                """
                self.wfile.write(success_html.encode("utf-8"))
                
                # Signal the server to stop
                self.server.token_retrieved = True
                
            except Exception as e:
                print(f"[ERROR] Failed to generate session: {e}")
                self.send_response(500)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                error_html = f"<html><body><h1>Authentication Failed</h1><p>Error: {e}</p></body></html>"
                self.wfile.write(error_html.encode("utf-8"))
        else:
            self.send_response(400)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h1>Bad Request</h1><p>Missing request_token</p></body></html>")

def run_local_server(api_key, api_secret):
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, OAuthHandler)
    httpd.token_retrieved = False
    
    # Initialize Kite Connect and generate login url
    kite = KiteConnect(api_key=api_key)
    login_url = kite.login_url()
    
    print("\n" + "="*60)
    print("ZERODHA KITE OAUTH FLOW")
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
        print("ERROR: ZERODHA_API_KEY or ZERODHA_API_SECRET is missing in .env!")
        print("Please configure them in your .env file before running this script.")
        print("Required fields:")
        print("  - ZERODHA_API_KEY")
        print("  - ZERODHA_API_SECRET")
        print("!"*60 + "\n")
        sys.exit(1)
        
    run_local_server(api_key, api_secret)

if __name__ == '__main__':
    main()
