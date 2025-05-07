from app import create_app

app = create_app()

if __name__ == '__main__':
    # Get host and port from config, fall back to defaults
    host = app.config.get('HOST', '0.0.0.0')
    port = app.config.get('PORT', 5001)
    debug = app.config.get('DEBUG', False)
    
    print(f"Starting Flask app on {host}:{port} (Debug: {debug})")
    # Use waitress or gunicorn for production instead of app.run(debug=True)
    app.run(host=host, port=port, debug=debug)