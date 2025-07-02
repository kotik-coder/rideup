import dash_bootstrap_components as dbc
from dash import dcc, html
import plotly.graph_objects as go
import numpy as np

# Assuming map_helpers is available for _calculate_zoom,
# but it's part of map_visualization in our new structure
# To avoid circular dependency with map_visualization here,
# _create_initial_figure will be moved here and needs _calculate_zoom logic directly or passed.
# Let's put _calculate_zoom here as it's directly used for initial figure setup.

def _calculate_zoom(lats, lons):
    """
    Вычисляет уровень масштабирования на основе географического охвата.
    """
    # Assuming print_step is imported from map_helpers
    from map_helpers import print_step

    if not lats or not lons:
        print_step("Zoom", "Расчет зума: Нет координат. Возвращаю дефолтный зум.")
        return 12

    lat_span = max(lats) - min(lats)
    lon_span = max(lons) - min(lons)

    if lat_span == 0 and lon_span == 0:
        print_step("Zoom", "Расчет зума: Нулевой охват. Возвращаю высокий зум.")
        return 15

    # These constants are empirically derived for OpenStreetMap tiles
    # Adjust as needed for specific mapping requirements
    zoom_lat = 9.5 - np.log2(lat_span + 1e-6)
    zoom_lon = 9.5 - np.log2(lon_span + 1e-6)

    final_zoom = min(zoom_lat, zoom_lon, 18) # Cap max zoom to 18
    print_step("Zoom", f"Рассчитан зум: {final_zoom:.2f}")
    return final_zoom

def _add_forest_boundary_and_name_to_figure(fig, bounds):
    """Adds the forest boundary and name annotation to the Plotly figure."""
    min_lon_val, min_lat_val, max_lon_val, max_lat_val = bounds
    from map_helpers import print_step

    lons_boundary = [min_lon_val, max_lon_val, max_lon_val, min_lon_val, min_lon_val]
    lats_boundary = [min_lat_val, min_lat_val, max_lat_val, max_lat_val, min_lat_val]

    fig.add_trace(go.Scattermap(
        lat=lats_boundary,
        lon=lons_boundary,
        mode='lines',
        line=dict(width=3, color='blue'),
        hoverinfo='none',
        showlegend=False,
        name="Границы Битцевского леса"
    ))

    center_lat = (min_lat_val + max_lat_val) / 2
    center_lon = (min_lon_val + max_lon_val) / 2

    fig.add_annotation(
        x=center_lon,
        y=center_lat,
        text="Битцевский Парк",
        showarrow=False,
        font=dict(size=20, color="black", family="Arial, sans-serif"),
        yanchor="middle",
        xanchor="center",
        bgcolor="rgba(255, 255, 255, 0.7)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4
    )
    print_step("Map Drawing", "Добавлены границы и название леса.")


def create_initial_figure(bounds):
    """Creates the base Plotly map figure with centering and style."""
    min_lon_val, min_lat_val, max_lon_val, max_lat_val = bounds

    center_lat = (min_lat_val + max_lat_val) / 2
    center_lon = (min_lon_val + max_lon_val) / 2

    initial_zoom = _calculate_zoom([min_lat_val, max_lat_val], [min_lon_val, max_lon_val])

    fig = go.Figure(go.Scattermap())
    fig.update_layout(
        map_style="open-street-map",
        map=dict(
            center=dict(lat=center_lat, lon=center_lon),
            zoom=initial_zoom
        ),
        margin={"r":0,"t":0,"l":0,"b":0},
        showlegend=False,
        clickmode='event+select'
    )

    _add_forest_boundary_and_name_to_figure(fig, bounds)

    return fig

def setup_layout(initial_bounds):
    """
    Sets up the main Dash application layout.
    Requires initial_bounds from RouteManager to create the initial map figure.
    """
    bottom_graphs_vh = 25
    top_row_dynamic_height = f'calc(100vh - {bottom_graphs_vh}vh - 15px)'

    layout = dbc.Container([
        dbc.Row([
            dbc.Col(dcc.Graph(
                id='map-graph',
                style={'height': '100%', 'width': '100%'},
                figure=create_initial_figure(initial_bounds)
            ), width=7, style={'padding': '0', 'height': '100%'}),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Выбор маршрута", className="font-weight-bold py-2"),
                    dbc.CardBody([
                        dcc.Dropdown(
                            id='route-selector',
                            options=[],
                            placeholder="Выберите маршрут",
                            className="mb-2"
                        )
                    ], className="py-2")
                ], className="mb-2"),
                dbc.Card([
                    dbc.CardHeader("Информация о маршруте", className="font-weight-bold py-2"),
                    dbc.CardBody([
                        html.Div(id='route-general-info')
                    ], className="py-2")
                ], className="mb-2"),
                dbc.Card([
                    dbc.CardHeader("Информация о чекпоинте", className="font-weight-bold py-2"),
                    dbc.CardBody(id='checkpoint-info', className="py-2")
                ])
            ], width=5, style={'padding': '0 15px', 'height': '100%', 'overflowY': 'auto'})
        ], style={'margin': '0', 'height': top_row_dynamic_height}),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='elevation-profile', style={'height': '100%', 'width': '100%'}),
            ], width=6),
            dbc.Col([
                dcc.Graph(id='velocity-profile', style={'height': '100%', 'width': '100%'})
            ], width=6)
        ], style={'margin': '0', 'marginTop': '15px', 'height': f'{bottom_graphs_vh}vh'}),
        dcc.Store(id='route-data-store'),
        dcc.Store(id='selected-route-index'),
        dcc.Store(id='selected-checkpoint-index'),
        html.Div(id='initial-load-trigger', style={'display': 'none'})
    ], fluid=True, style={'height': '100vh', 'padding': '0', 'overflowY': 'hidden', 'overflowX': 'hidden'})

    return layout