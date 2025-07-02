import dash_bootstrap_components as dbc
from dash import html

def create_checkpoint_card(checkpoint):
    """
    Generates a Dash Bootstrap Components card for displaying checkpoint information.
    """
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H5(f"{checkpoint['name']} ({checkpoint['position']}/{checkpoint['total_positions']})",
                       className="card-title mb-1 fs-6"),
                html.P([html.B("Координаты: "), f"{checkpoint['lat']:.5f}, {checkpoint['lon']:.5f}"], className="mb-0", style={'fontSize': '0.9em'}),
                html.P([html.B("Высота: "), f"{checkpoint['elevation']:.1f} м"], className="mb-0", style={'fontSize': '0.9em'}),
                html.P([html.B("Расстояние от старта: "), f"{checkpoint.get('distance_from_start', 0):.1f} м"], className="mb-0", style={'fontSize': '0.9em'}),
                html.P(html.Em(checkpoint['description']), className="card-text mb-0", style={'fontSize': '0.9em'})
            ], width=6),
            dbc.Col([
                html.Iframe(
                    srcDoc=checkpoint['photo_html'],
                    style={'width': '100%', 'height': '220px', 'border': 'none', 'marginBottom': '5px'}
                )
            ], width=6)
        ], className="g-0"),
        dbc.Row([
            dbc.Col(dbc.Button("← Предыдущий", id="prev-checkpoint-button", className="me-2", color="secondary", size="sm"), width={"size": 6, "offset": 0}),
            dbc.Col(dbc.Button("Следующий →", id="next-checkpoint-button", className="ms-2", color="secondary", size="sm"), width={"size": 6, "offset": 0})
        ], className="mt-2 text-center")
    ])