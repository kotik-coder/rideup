# app_core.py
import os
import dash
import dash_bootstrap_components as dbc
from flask import send_from_directory
import logging

from src.ui.callbacks import setup_callbacks
from src.routes.spot import Spot
from src.routes.route_processor import RouteProcessor
from src.ui.layout import setup_layout
from src.ui.map_helpers import DEBUG, print_step

class BitsevskyMapApp:
    
    spot : Spot
    route_processor : RouteProcessor
        
    def _configure_server(self):
        assets_path = os.getcwd() + '/assets'
        
        # Create single Dash instance with all configurations
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP, "https://use.fontawesome.com/releases/v5.15.4/css/all.css"],
            suppress_callback_exceptions=True,
            assets_folder=assets_path,
        )
        
        # Disable Dash devtools verbose logging
        if not DEBUG:
            log = logging.getLogger('werkzeug')
            log.setLevel(logging.ERROR)
            self.app.logger.setLevel(logging.WARNING)
    
    def __init__(self):
        self._configure_server()

        self._print_header()
        
        #init backend
        self._init_spot()
        
        #set up layout for ui
        self.app.layout = setup_layout(spot=self.spot)
        
        # Setup callbacks after everything is initialized
        setup_callbacks(
            self.app,
            spot=self.spot,
            route_processor=self.route_processor
        )
        
    def _init_spot(self):
        print_step("Core", "Creating spot")
        # Initialize Spot with proper parameters
        self.spot = Spot(
            geostring="Битцевский лес, Москва",
            weather_api_key="18e37b7ae267bb45bfca6676e7ae072f"
        )                
        
        print_step("Core", "Creating route processor")
        # Initialize RouteProcessor with loaded data
        self.route_processor = RouteProcessor(
            local_photos=self.spot.local_photos,
            all_tracks=self.spot.tracks
        )

    def _print_header(self):
        """Prints startup header information"""
        from datetime import datetime
        print("\n" + "="*50)
        print(f"Loading Bitsevsky Forest Map")
        print(f"Start time: {datetime.now().strftime('%H:%M:%S')}")
        print("="*50 + "\n")

    def run(self):
        self.app.run(debug=DEBUG)

if __name__ == '__main__':
    app = BitsevskyMapApp()
    app.run()
