# app_core.py
import os
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from flask import send_from_directory
import json
import logging
from typing import List

from spot import Spot, SpotLoader
from route_processor import RouteProcessor, ProcessedRoute
from layout import setup_layout
from callbacks import setup_callbacks
from map_helpers import DEBUG, print_step

class BitsevskyMapApp:
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
        
        print_step("Core", "Creating spot")
        # Initialize Spot with proper parameters
        self.spot = Spot(
            geostring="Битцевский лес, Москва",
            local_photos_folder=local_photos_dir
        )
        
        print_step("Core", "Initi'ing spot loader")
        # Initialize SpotLoader and load data
        self.spot_loader = SpotLoader(self.spot)
        self.spot_loader.load_valid_routes_and_tracks()
        
        print_step("Core", "Creating route processor")
        # Initialize RouteProcessor with loaded data
        self.route_processor = RouteProcessor(
            local_photos=self.spot.local_photos,
            all_tracks=self.spot.tracks
        )

        print_step("Core", "Processing initial routes")
        # Process initial routes
        self.processed_routes: List[ProcessedRoute] = [
            self.route_processor.process_route(route)
            for route in self.spot.routes
        ]

        self.selected_route_index = 0 if self.processed_routes else None
        self.selected_checkpoint_index = None

        print_step("Core", "Setting up UI layout")
        # Setup layout with spot and initial routes
        self.app.layout = setup_layout(
            spot=self.spot,
            routes=self.processed_routes
        )
        
        print_step("Core", "Setting up callbacks")
        # Setup callbacks with necessary components
        setup_callbacks(
            app=self.app,
            spot=self.spot,
            spot_loader=self.spot_loader,
            route_processor=self.route_processor
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