# rideup

Проект для визуализации велосипедного маршрута в Битцевском лесу (Москва) с возможностью переключения между схематичной и спутниковой картами.

## Особенности

- Автоматическая генерация маршрута по заданным правилам:
  - Старт и финиш на границе леса
  - Расстояние между точками ≤500 м
  - Финиш в пределах 500 м от последней точки
- Визуализация маршрута с цветовым градиентом
- Два режима отображения карты:
  - Схема (OpenStreetMap)
  - Спутниковые снимки (ArcGIS)
- Серые области за пределами зоны маршрута

## Требования

- Python 3.7+
- Установленные пакеты:
  ```
  pip install PyQt5 folium osmnx shapely branca
  ```

## Запуск

```bash
python map.py
```

## Лицензия

Apache License 2.0

```
Copyright 2023 Artem Lunev

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
