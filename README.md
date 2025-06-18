# Deep-Kalman-Fun

### Описание

Данный проект предназначен для исследования и практического применения методов фильтра Калмана и нейросетевых фильтров (KalmanNet, DANSE) в задаче сглаживания предсказаний позы человека на видео. Проект реализован в виде Google Colab-ноутбука и Python-модулей, что обеспечивает простоту запуска и модификации.

### Основные возможности

- **Детектирование позы**: автоматическое извлечение ключевых точек с помощью YOLO Pose.
- **Сглаживание траекторий**: применение классических и нейросетевых фильтров к координатам ключевых точек.
- **Генерация синтетических данных**: создание контролируемых траекторий с шумом для обучения и тестирования.
- **Обучение KalmanNet и DANSE**: гибкая настройка архитектур и параметров, сохранение и загрузка весов моделей.
- **Сравнение методов**: автоматический расчет метрик качества и построение сравнительных таблиц.
- **Визуализация**: построение графиков траекторий и сохранение видео с наложением позы для каждого метода.


### Установка

1. Клонируйте репозиторий или загрузите Colab-ноутбук.
2. Установите зависимости:

```bash
pip install ultralytics torch torchvision filterpy opencv-python-headless matplotlib
```

3. (Опционально) Скачайте веса обученных моделей KalmanNet и DANSE.

### Быстрый старт

1. Запустите ноутбук в Google Colab.
2. Укажите путь к вашему видео и настройте параметры модели YOLO Pose.
3. Запустите блоки кода для:
    - Детектирования позы на видео.
    - Применения фильтров (Kalman, Moving Average, KalmanNet, DANSE).
    - Визуализации и сохранения результатов.

### Пример использования

```python
from ultralytics import YOLO
from kalman_pose_filtering import apply_kalman_filter, KalmanNet, DANSE

# Детектирование позы
model = YOLO('yolov8n-pose.pt')
keypoints = detect_pose_yolo_fixed('your_video.mp4', model)

# Сглаживание
kalman_filtered = apply_kalman_filter(keypoints)
kalmannet_model = KalmanNet()
kalmannet_model.load_state_dict(torch.load('kalmannet_model.pth'))
kalmannet_filtered = kalmannet_model(torch.tensor(keypoints[None, :, :]).float())[^0][^0].numpy()
```


### Метрики оценки

- **MAE** — средняя абсолютная ошибка.
- **RMSE** — среднеквадратичная ошибка.
- **VelocityError** — ошибка по скорости.
- **AccelerationError** — ошибка по ускорению.
- **Jitter** — индекс дрожания.
- **Smoothness** — оценка сглаженности.
- **PCK** — процент корректных ключевых точек.

Формулы и подробное описание приведены в документации.

### Визуализация

- Графики траекторий ключевых точек для разных методов.
- Сохранение видео с наложением позы для каждого метода (цветовая дифференциация).


### Ссылки

- [KalmanNet: Neural Network Kalman Filtering](https://arxiv.org/abs/2107.10043)
- [DANSE: Data-driven Nonlinear State Estimation](https://arxiv.org/abs/2410.12289)
- [AI-Aided Kalman Filters](https://arxiv.org/abs/2410.12289)

---
