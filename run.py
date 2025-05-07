from app import create_app

# Gunicorn will look for this 'app' object at the module level.
app = create_app()

if __name__ == '__main__':
    # This block is for direct execution (e.g., local development)
    # and is not used by Gunicorn in production.
    # Get host and port from app config, fall back to defaults
    host = app.config.get('HOST', '0.0.0.0')
    port = app.config.get('PORT', 5001) # Default port for local dev
    debug = app.config.get('DEBUG', False)
    
    print(f"Starting Flask development server on http://{host}:{port} (Debug: {debug})")
    app.run(host=host, port=port, debug=debug)