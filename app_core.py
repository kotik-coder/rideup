import os
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from flask import send_from_directory
import json
import logging

from spot import Spot, SpotLoader
from route_processor import RouteProcessor
from layout import setup_layout
from callbacks import setup_callbacks

class BitsevskyMapApp:
    def __init__(self):
        # Configure Dash to be quieter
        self.app = dash.Dash(__name__,
                            external_stylesheets=[dbc.themes.BOOTSTRAP],
                            suppress_callback_exceptions=True)

        # Disable Dash devtools verbose logging
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
        
        # Initialize Spot and SpotLoader instead of RouteManager
        self.spot = Spot("Битцевский лес, Москва")
        self.spot_loader = SpotLoader(self.spot)
        self.route_processor = RouteProcessor([], {})  # Will be populated later
        
        self.selected_route_index = None
        self.selected_checkpoint_index = None

        # Setup layout with spot bounds
        self.app.layout = setup_layout(self.spot.bounds)
        
        # Setup callbacks with necessary components
        setup_callbacks(self.app, self.spot, self.spot_loader, self.route_processor)

    def _print_header(self):
        """Выводит заголовок с информацией о запуске"""
        from datetime import datetime
        print("\n" + "="*50)
        print(f"Загрузка карты Битцевского леса")
        print(f"Время начала: {datetime.now().strftime('%H:%M:%S')}")
        print("="*50 + "\n")

    def run(self):
        self.app.run(debug=False)

if __name__ == '__main__':
    app = BitsevskyMapApp()
    app.run()