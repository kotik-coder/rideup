# app_core.py
import os
import dash
import dash_bootstrap_components as dbc
from flask import send_from_directory
import logging

from callbacks import setup_callbacks
from map_visualization import create_base_map
from spot import Spot, SpotLoader
from route_processor import RouteProcessor
from layout import setup_layout
from map_helpers import DEBUG, print_step

class BitsevskyMapApp:
    
    spot : Spot
    spot_loader : SpotLoader
    route_processor : RouteProcessor
    
    def __init__(self):
        # Configure Dash to be quieter
        self.app = dash.Dash(__name__,
                          external_stylesheets=[dbc.themes.BOOTSTRAP],
                          suppress_callback_exceptions=True)

        # Disable Dash devtools verbose logging
        if not DEBUG:
            log = logging.getLogger('werkzeug')
            log.setLevel(logging.ERROR)
            self.app.logger.setLevel(logging.WARNING)

        local_photos_dir = os.path.join(os.path.dirname(__file__), 'local_photos')

        self.app.server.add_url_rule(
            '/local_photos/<path:filename>',
            endpoint='local_photos',
            view_func=lambda filename: send_from_directory(local_photos_dir, filename)
        )

        self._print_header()
        
        #init backend
        self._init_spot(local_photos_dir)
        
        #set up layout for ui
        self.app.layout = setup_layout(spot=self.spot)
        
        # Setup callbacks after everything is initialized
        setup_callbacks(
            self.app,
            spot=self.spot,
            spot_loader=self.spot_loader,
            route_processor=self.route_processor
        )
        
    def _init_spot(self, local_photos_dir):        
        print_step("Core", "Creating spot")
        # Initialize Spot with proper parameters
        self.spot = Spot(
            geostring="Битцевский лес, Москва",
            local_photos_folder=local_photos_dir
        )
        
        print_step("Core", "Initi'ing spot loader")
        
        # Initialize SpotLoader and load data for the selected spot
        self.spot_loader = SpotLoader(self.spot)
        self.spot_loader.load_valid_routes_and_tracks()
        
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
        self.app.run(debug=False)

if __name__ == '__main__':
    app = BitsevskyMapApp()
    app.run()