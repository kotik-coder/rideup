import os
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from flask import send_from_directory
import json
import logging # Import logging

from route_manager import RouteManager
from map_helpers import print_step # Using existing map_helpers for print_step
from layout import setup_layout, create_initial_figure # Import layout setup and initial figure
from callbacks import setup_callbacks # Import all callbacks
from ui_components import create_checkpoint_card # For the checkpoint card method initially in BitsevskyMapApp

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
        print(f"DEBUG: local_photos_dir is set to: {local_photos_dir}")

        self.app.server.add_url_rule(
            '/local_photos/<path:filename>',
            endpoint='local_photos',
            view_func=lambda filename: send_from_directory(local_photos_dir, filename)
        )

        self._print_header()
        self.rm = RouteManager("Битцевский лес, Москва")
        self.selected_route_index = None
        self.selected_checkpoint_index = None

        # Pass self.rm and shared state to layout and callbacks
        # Layout needs rm.bounds for initial figure
        self.app.layout = setup_layout(self.rm.bounds)

        # Callbacks need access to self.rm and potentially methods of this class
        # Pass necessary components for callbacks to function
        setup_callbacks(self.app, self.rm)

        # Expose the _create_checkpoint_card method from ui_components via the app instance
        # This is a bit of a workaround if callbacks need a method from another module,
        # but prevents circular dependency if ui_components were to import app_core
        # Alternatively, ui_components could be a standalone function that takes data.
        # For now, let's keep it clean by ensuring callbacks don't directly call
        # methods of BitsevskyMapApp, but rather functions imported from other modules.
        # The create_checkpoint_card is called directly from update_checkpoint_info in callbacks.py
        # so this line is no longer necessary here.

    def _print_header(self):
        """Выводит заголовок с информацией о запуске"""
        from datetime import datetime # Local import to avoid circular dependency with other modules that might use datetime
        print("\n" + "="*50)
        print(f"Загрузка карты Битцевского леса")
        print(f"Время начала: {datetime.now().strftime('%H:%M:%S')}")
        print("="*50 + "\n")

    def run(self):
        self.app.run(debug=False)

if __name__ == '__main__':
    app = BitsevskyMapApp()
    app.run()