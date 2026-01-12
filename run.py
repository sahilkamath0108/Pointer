import os
import asyncio
from dotenv import load_dotenv

async def start_server():
    load_dotenv()
    
    # Import here to ensure environment variables are loaded first
    from app import app
    
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting waWeb server on port {port}...")
    print(f"Webhook URL: http://localhost:{port}/webhook")
    print("Press Ctrl+C to stop the server.")
    
    # Start the Flask development server
    app.run(host='0.0.0.0', port=port, debug=True)

if __name__ == "__main__":
    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except Exception as e:
        print(f"Error starting server: {str(e)}") 