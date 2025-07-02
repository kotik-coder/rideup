import plotly.graph_objects as go
import numpy as np
from map_helpers import print_step
from route import GeoPoint # For bearing calculation


def calculate_zoom(lats, lons):
    """
    Вычисляет уровень масштабирования на основе географического охвата.
    This is a copy of the function in layout.py to avoid circular imports.
    If this function is only used by update_map_figure, it belongs here.
    If layout.py also needs it directly (which it does for initial figure),
    it might be better to have it in a shared 'geometry_utils.py' or similar.
    For now, duplicating to avoid circular imports with layout.py/callbacks.py.
    """
    if not lats or not lons:
        print_step("Zoom", "Расчет зума: Нет координат. Возвращаю дефолтный зум.")
        return 12

    lat_span = max(lats) - min(lats)
    lon_span = max(lons) - min(lons)

    if lat_span == 0 and lon_span == 0:
        print_step("Zoom", "Расчет зума: Нулевой охват. Возвращаю высокий зум.")
        return 15

    zoom_lat = 9.5 - np.log2(lat_span + 1e-6)
    zoom_lon = 9.5 - np.log2(lon_span + 1e-6)

    final_zoom = min(zoom_lat, zoom_lon, 18)
    print_step("Zoom", f"Рассчитан зум: {final_zoom:.2f}")
    return final_zoom

def add_forest_boundary_and_name_to_figure(fig, bounds):
    """
    Adds the forest boundary and name annotation to the Plotly figure.
    This is a copy of the function in layout.py to avoid circular imports.
    Similar to calculate_zoom, if used by both, a shared geometry_utils might be best.
    """
    min_lon_val, min_lat_val, max_lon_val, max_lat_val = bounds

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


def add_route_to_figure(fig, route_dict, is_selected=False, highlight_checkpoint=None):
    """Adds a route trace to the Plotly figure."""
    route_lats = [p[0] for p in route_dict['smooth_points']]
    route_lons = [p[1] for p in route_dict['smooth_points']]
    route_elevations = route_dict['elevation_profile']

    if not route_lats or not route_lons or not route_elevations:
        print_step("Map Drawing", f"Пропускаю отрисовку маршрута '{route_dict.get('name', 'N/A')}' из-за отсутствия данных.")
        return

    if is_selected:
        elevations_only = [ep['elevation'] for ep in route_elevations]
        if elevations_only:
            min_elev = min(elevations_only)
            max_elev = max(elevations_only)
            start_elev = elevations_only[0] if elevations_only else 0

            if min_elev == max_elev:
                plotly_colorscale = [[0, 'green'], [1, 'green']]
                cmin_val = min_elev - 1 if min_elev > 0 else 0
                cmax_val = max_elev + 1
            else:
                norm_start_elev = (start_elev - min_elev) / (max_elev - min_elev)
                plotly_colorscale = [
                    [0, 'blue'],
                    [norm_start_elev, 'green'],
                    [1, 'red']
                ]
                plotly_colorscale.sort(key=lambda x: x[0])
                cmin_val = min_elev
                cmax_val = max_elev

            fig.add_trace(go.Scattermap(
                lat=route_lats,
                lon=route_lons,
                mode='lines+markers',
                line=dict(width=6, color='rgba(0,0,0,0)'),
                marker=dict(
                    size=12,
                    color=elevations_only,
                    colorscale=plotly_colorscale,
                    cmin=cmin_val,
                    cmax=cmax_val,
                    colorbar=dict(
                        title="Высота (м)",
                        x=1.02,
                        lenmode="fraction",
                        len=0.75
                    )
                ),
                hoverinfo='text',
                hovertext=[f"Высота: {ep['elevation']:.1f} м" for ep in route_elevations],
                showlegend=False,
                name=f"Маршрут {route_dict['name']} (по высоте)"
            ))
            print_step("Map Drawing", f"Отрисован выбранный маршрут '{route_dict['name']}' с цветовой схемой высот.")
        else:
            print_step("Map Drawing", f"Пропускаю отрисовку выбранного маршрута '{route_dict['name']}' по высоте: нет данных высот.")
            fig.add_trace(go.Scattermap(
                lat=route_lats,
                lon=route_lons,
                mode='lines',
                line=dict(width=6, color='gray'),
                hoverinfo='text',
                hovertext=[f"Маршрут: {route_dict['name']}" for _ in route_lats],
                showlegend=False,
                name=f"Маршрут {route_dict['name']}"
            ))

        if route_dict['checkpoints']:
            checkpoint_lats = [cp['lat'] for cp in route_dict['checkpoints']]
            checkpoint_lons = [cp['lon'] for cp in route_dict['checkpoints']]
            checkpoint_names = [cp['name'] for cp in route_dict['checkpoints']]
            checkpoint_elevations = [cp['elevation'] for cp in route_dict['checkpoints']]
            checkpoint_indices = [idx for idx, cp in enumerate(route_dict['checkpoints'])]

            fig.add_trace(go.Scattermap(
                lat=checkpoint_lats,
                lon=checkpoint_lons,
                mode='markers',
                marker=dict(
                    size=16,
                    symbol='circle',
                    color='lime',
                    opacity=1.0
                ),
                text=checkpoint_names,
                hoverinfo='text',
                hovertext=[f"{name}<br>Высота: {elev:.1f} м" for name, elev in zip(checkpoint_names, checkpoint_elevations)],
                customdata=checkpoint_indices,
                showlegend=False,
                name="Чекпоинты"
            ))
            print_step("Map Drawing", f"Отрисованы чекпоинты (теперь как круги для отладки) для выбранного маршрута '{route_dict['name']}'.")

        if highlight_checkpoint is not None and highlight_checkpoint < len(route_dict['checkpoints']):
            checkpoint = route_dict['checkpoints'][highlight_checkpoint]

            fig.add_trace(go.Scattermap(
                lat=[checkpoint['lat']],
                lon=[checkpoint['lon']],
                mode='markers',
                marker=dict(
                    size=22,
                    symbol='circle',
                    color='red',
                    opacity=1.0,
                ),
                name="Выбранный чекпоинт",
                hoverinfo='text',
                hovertext=f"{checkpoint['name']}<br>Высота: {checkpoint['elevation']:.1f} м",
                showlegend=False
            ))
            print_step("Map Drawing", f"Отрисован выделенный чекпоинт {highlight_checkpoint} (теперь как круг для отладки) на маршруте '{route_dict['name']}'.")

    else:
        fig.add_trace(go.Scattermap(
            lat=route_lats,
            lon=route_lons,
            mode='lines',
            line=dict(width=3, color='rgba(100, 100, 100, 0.5)'),
            hoverinfo='text',
            hovertext=f"Маршрут: {route_dict['name']}",
            showlegend=False,
            name=route_dict['name']
        ))
        print_step("Map Drawing", f"Отрисован невыбранный маршрут '{route_dict['name']}'.")