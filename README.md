# NTO AI 2025-2026: Baseline для рекомендательной системы

Baseline-решение для задачи предсказания оценок книг пользователями в рамках соревнования НТО 2025/2026 (Профиль «Искусственный интеллект»).

## Описание

Модель предсказывает оценку (rating) от 0 до 10, которую пользователь поставит книге. Решение основано на:
- **LightGBM** с 5-fold GroupKFold cross-validation
- **Feature Engineering**:
  - Агрегированные признаки: средние оценки и количество оценок для пользователей, книг и авторов
  - Метаданные: пол, возраст, год публикации, язык, издательство
  - Жанры: количество жанров для каждой книги
- **Текстовые признаки**:
  - **TF-IDF**: 500 фичей из текстовых описаний книг (биграммы)
  - **BERT embeddings**: 768-мерные эмбеддинги из русскоязычной BERT-модели (`DeepPavlov/rubert-base-cased`)

## Быстрый старт

### Установка зависимостей

```bash
poetry install
```

### Подготовка данных

Поместите CSV-файлы в `data/raw/`:
- `stage1_public_train.csv`
- `stage1_public_test.csv`
- `stage1_public_users.csv`
- `stage1_public_books.csv`
- `stage1_public_book_genres.csv`
- `stage1_public_genres.csv`
- `stage1_public_book_descriptions.csv`

### Запуск пайплайна

```bash
# Обучение
poetry run python -m src.baseline.train

# Предсказание
poetry run python -m src.baseline.predict

# Валидация submission
poetry run python -m src.baseline.validate
```

Или через Makefile:
```bash
make train    # Обучение
make predict  # Предсказание
make validate # Валидация
make run      # Полный цикл
```

## Структура проекта

```
.
├── data/
│   └── raw/              # Исходные CSV-файлы
├── output/
│   ├── models/           # Обученные модели и TF-IDF векторайзер
│   └── submissions/      # Файлы submission
├── src/baseline/
│   ├── config.py         # Конфигурация и параметры модели
│   ├── constants.py      # Константы проекта (имена файлов, колонок)
│   ├── data_processing.py # Загрузка и объединение данных
│   ├── features.py       # Feature engineering (агрегаты, жанры, TF-IDF, BERT)
│   ├── train.py          # Обучение модели
│   ├── predict.py        # Генерация предсказаний
│   ├── validate.py       # Проверка формата submission
│   └── evaluate.py       # Оценка качества предсказаний (метрики)
└── Makefile              # Удобные команды
```

## Особенности реализации

- **Предотвращение data leakage**: TF-IDF векторайзер и BERT эмбеддинги вычисляются только на train данных
- **GroupKFold**: Разбиение по `user_id` для корректной валидации (избегаем утечки данных между пользователями)
- **Обработка пропусков**: Автоматическое заполнение для всех признаков (глобальное среднее для агрегатов, медиана для возраста)
- **Кэширование**: BERT эмбеддинги сохраняются на диск для ускорения последующих запусков
- **Типизация**: Полная типизация кода (type hints)
- **Code quality**: Ruff для линтинга и форматирования, pre-commit hooks

## Оценка модели

Для оценки качества предсказаний используется скрипт `evaluate.py`:

```bash
poetry run python -m src.baseline.evaluate --solution stage1_private_solution.csv --submission output/submissions/submission.csv
```

Скрипт вычисляет метрики для public и private частей тестовой выборки.

## Метрика

Score рассчитывается на основе RMSE и MAE:
```
Score = 1 - (0.5 * RMSE/10 + 0.5 * MAE/10)
```

Предсказания автоматически ограничиваются диапазоном [0, 10].

## Зависимости

- Python >= 3.10
- pandas, scikit-learn, lightgbm, joblib
- transformers, torch, sentencepiece (для BERT эмбеддингов)
- ruff, pre-commit (dev)

## Лицензия

Проект создан для соревнования НТО 2025-2026.

