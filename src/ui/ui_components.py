import dash_bootstrap_components as dbc
from dash import html
import numpy as np

from src.routes.route import Route
from src.routes.checkpoints import Checkpoint, PhotoCheckpoint
from src.routes.route_processor import ProcessedRoute

def create_route_info_card(route : Route, processed_route : ProcessedRoute):
    """
    Generates a Dash Bootstrap Components card for displaying route information.
    
    Args:
        processed_route: ProcessedRoute object containing route information
        
    Returns:
        html.Div containing the route information card
    """
    return html.Div([
        html.H5(route.name, className="mb-1 fs-6"),
        html.P(f"Длина маршрута: {route.total_distance:.2f} м", 
              className="mb-1", style={'fontSize': '0.9em'}),
        html.P(f"Средняя высота: {np.mean([e for e in route.elevations]):.1f} м", 
              className="mb-1", style={'fontSize': '0.9em'}),
        html.P(f"Набор высоты: {sum(s.elevation_gain for s in processed_route.segments):.1f} м", 
              className="mb-1", style={'fontSize': '0.9em'}),
        html.P(f"Потеря высоты: {sum(s.elevation_loss for s in processed_route.segments):.1f} м", 
              className="mb-0", style={'fontSize': '0.9em'}),
    ])

def create_checkpoint_card(checkpoint, route : ProcessedRoute):
    """
    Generates a Dash Bootstrap Components card for displaying checkpoint information.
    Now accepts a Checkpoint object instead of a dictionary.
    """
    
    optional_block = dbc.Col(width=0)
    
    if checkpoint.photo_html:
        optional_block = dbc.Col([
                html.Iframe(
                    srcDoc=checkpoint.photo_html,
                    style={'width': '100%', 'height': '220px', 'border': 'none', 'marginBottom': '5px'}
                )
            ], width=6)
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H5(f"{checkpoint.name}",
                       className="card-title mb-1 fs-6"),
                html.P([html.B("Координаты: "), f"{checkpoint.point.lat:.5f}, {checkpoint.point.lon:.5f}"], className="mb-0", style={'fontSize': '0.9em'}),
                html.P([html.B("Высота: "), f"{checkpoint.point.elevation:.1f} м"], className="mb-0", style={'fontSize': '0.9em'}),
                html.P([html.B("Расстояние от старта: "), f"{route.smooth_points[0].distance_to(checkpoint.point):.1f} м"], className="mb-0", style={'fontSize': '0.9em'}),
                html.P(html.Em(checkpoint.description), className="card-text mb-0", style={'fontSize': '0.9em'})
            ], width=6),
            optional_block
        ], className="g-0"),
        dbc.Row([
            dbc.Col(dbc.Button("← Предыдущий", id="prev-checkpoint-button", className="me-2", color="secondary", size="sm"), width={"size": 6, "offset": 0}),
            dbc.Col(dbc.Button("Следующий →",  id="next-checkpoint-button", className="ms-2", color="secondary", size="sm"), width={"size": 6, "offset": 0})
        ], className="mt-2 text-center")
    ])
    
    