#!/usr/bin/env python3
"""
Лабораторная работа №2 - УПРОЩЕННАЯ ВЕРСИЯ
Создание графиков декомпозиции и таблиц для отчета
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.fft import fft, fftfreq

# ===================================================================
# КОНФИГУРАЦИОННЫЕ ПАРАМЕТРЫ
# ===================================================================

# Пути к файлам
INPUT_FILE = "out/clean.xlsx"
OUTPUT_DIR = "out/lab2"
INPUT_SHEET = "clean"
DATE_COLUMN = "data"
VALUE_COLUMN = "curs"

# Параметры декомпозиции
MAX_POLY_DEGREE = 3          # Максимальная степень полиномиального тренда
N_HARMONICS = 5              # Количество гармоник для Фурье-анализа

# Параметры Prophet
PROPHET_DAILY_SEASONALITY = False
PROPHET_WEEKLY_SEASONALITY = True
PROPHET_YEARLY_SEASONALITY = True
PROPHET_SEASONALITY_MODE = 'additive'

# Проценты данных для анализа
DECOMPOSITION_PERCENTAGES = [25, 50, 75, 90]

# Параметры графиков
FIGURE_DPI = 150
FIGURE_SIZE_MAIN = (15, 10)

# ===================================================================

# Полностью отключаем все предупреждения
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

# Дополнительное подавление специфичных предупреждений
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Импорты для моделей временных рядов
from statsmodels.stats.stattools import jarque_bera
import prophet
from prophet import Prophet

# Настройка стиля графиков
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def read_excel_series(path: str, sheet: str, date_col: str, value_col: str) -> pd.Series:
    """Загрузить Excel и вернуть ряд со временным индексом."""
    df = pd.read_excel(path, sheet_name=sheet)
    dates = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    values = pd.to_numeric(df[value_col], errors="coerce")
    ser = pd.Series(values.values, index=dates)
    ser = ser.sort_index().dropna()
    return ser

def fourier_decomposition(series: pd.Series, n_harmonics: int = N_HARMONICS) -> dict:
    """Декомпозиция с использованием Фурье-анализа (многочлены Чебышева)"""
    try:
        # Подготовка данных
        values = series.values
        n = len(values)
        t = np.arange(n)
        
        # Полиномиальный тренд (до 3-й степени)
        best_trend = None
        best_r2 = -1
        best_degree = 1
        
        for degree in range(1, MAX_POLY_DEGREE + 1):
            poly_features = PolynomialFeatures(degree=degree)
            t_poly = poly_features.fit_transform(t.reshape(-1, 1))
            trend_model = LinearRegression()
            trend_model.fit(t_poly, values)
            trend = trend_model.predict(t_poly)
            r2 = r2_score(values, trend)
            
            if r2 > best_r2:
                best_r2 = r2
                best_trend = trend
                best_degree = degree
        
        print(f"    Выбран полиномиальный тренд степени {best_degree} (R2 = {best_r2:.4f})")
        
        # Удаляем тренд
        detrended = values - best_trend
        
        # Фурье-анализ для сезонности
        fft_values = fft(detrended)
        freqs = fftfreq(n)
        
        # Находим доминирующие частоты
        power = np.abs(fft_values) ** 2
        dominant_freqs_idx = np.argsort(power)[-n_harmonics-1:-1]  # Исключаем DC компоненту
        
        # Строим сезонную компоненту
        seasonal = np.zeros(n)
        for idx in dominant_freqs_idx:
            if freqs[idx] != 0:  # Исключаем DC компоненту
                freq = freqs[idx]
                amplitude = np.abs(fft_values[idx]) / n * 2
                phase = np.angle(fft_values[idx])
                seasonal += amplitude * np.cos(2 * np.pi * freq * t + phase)
        
        # Остатки
        residuals = detrended - seasonal
        
        # Метрики качества
        reconstructed = best_trend + seasonal
        rmse = np.sqrt(mean_squared_error(values, reconstructed))
        mae = mean_absolute_error(values, reconstructed)
        r2_total = r2_score(values, reconstructed)
        
        return {
            'trend': best_trend,
            'seasonal': seasonal,
            'residuals': residuals,
            'reconstructed': reconstructed,
            'trend_degree': best_degree,
            'n_harmonics': n_harmonics,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2_total,
            'values': values  # Добавляем исходные значения для проверки
        }
        
    except Exception as e:
        print(f"    Ошибка в декомпозиции: {e}")
        return {'error': str(e)}

def build_prophet_model(series: pd.Series) -> dict:
    """Построение модели Prophet с декомпозицией"""
    try:
        # Подготовка данных для Prophet
        df = pd.DataFrame({
            'ds': series.index,
            'y': series.values
        })
        
        # Создание и обучение модели
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Prophet(
                daily_seasonality=PROPHET_DAILY_SEASONALITY,
                weekly_seasonality=PROPHET_WEEKLY_SEASONALITY,
                yearly_seasonality=PROPHET_YEARLY_SEASONALITY,
                seasonality_mode=PROPHET_SEASONALITY_MODE
            )
            model.fit(df)
        
        # Прогноз на тех же данных для получения компонентов
        forecast = model.predict(df)
        
        # Извлекаем компоненты
        trend = forecast['trend'].values
        # Суммируем все сезонные компоненты
        seasonal_components = []
        if 'yearly' in forecast.columns:
            seasonal_components.append(forecast['yearly'].values)
        if 'weekly' in forecast.columns:
            seasonal_components.append(forecast['weekly'].values)
        if 'daily' in forecast.columns:
            seasonal_components.append(forecast['daily'].values)
        
        if seasonal_components:
            seasonal = np.sum(seasonal_components, axis=0)
        else:
            seasonal = np.zeros(len(forecast))
        
        # Остатки
        residuals = df['y'].values - forecast['yhat'].values
        
        # Метрики качества
        rmse = np.sqrt(mean_squared_error(df['y'], forecast['yhat']))
        mae = mean_absolute_error(df['y'], forecast['yhat'])
        r2 = r2_score(df['y'], forecast['yhat'])
        
        return {
            'model': model,
            'forecast': forecast,
            'trend': trend,
            'seasonal': seasonal,
            'residuals': residuals,
            'reconstructed': forecast['yhat'].values,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'values': df['y'].values  # Добавляем исходные значения для проверки
        }
        
    except Exception as e:
        print(f"    Ошибка в Prophet: {e}")
        return {'error': str(e)}

def create_individual_chebyshev_plots(series: pd.Series, output_dir: str):
    """Создать отдельные графики декомпозиции Чебышева для каждого процента данных"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Вычисляем длины интервалов в процентах от общего количества данных
    total_length = len(series)
    interval_lengths = [int(total_length * p / 100) for p in DECOMPOSITION_PERCENTAGES]
    
    for length, percentage in zip(interval_lengths, DECOMPOSITION_PERCENTAGES):
        # Берем данные нужной длины
        if len(series) > length:
            test_series = series.iloc[:length]
        else:
            test_series = series
        
        print(f"    Чебышев {percentage}%: данные от {test_series.index[0]} до {test_series.index[-1]}")
        print(f"    Чебышев {percentage}%: значения от {test_series.values[0]:.2f} до {test_series.values[-1]:.2f}")
        
        # Выполняем декомпозицию
        decomp_results = fourier_decomposition(test_series)
        
        if 'error' not in decomp_results:
            x = range(len(test_series))
            
            # Создаем файл с 2 подграфиками для каждого процента
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            fig.suptitle(f'Декомпозиция временного ряда - {percentage}% данных (n={length})\nМногочлены Чебышева', 
                        fontsize=14, fontweight='bold')
            
            # График 1: Тренд + Исходные данные
            ax1.plot(x, test_series.values, 'b-', linewidth=2, label='Исходные данные', alpha=0.8)
            ax1.plot(x, decomp_results['trend'], 'r--', linewidth=2, label='Тренд', alpha=0.8)
            ax1.set_title(f'Тренд и исходные данные (RMSE: {decomp_results["rmse"]:.4f}, R²: {decomp_results["r2_score"]:.4f})')
            ax1.set_xlabel('Временной индекс')
            ax1.set_ylabel('Значение')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # График 2: Сезонность
            ax2.plot(x, decomp_results['seasonal'], 'g-', linewidth=2, label='Сезонная компонента', alpha=0.8)
            ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3, label='Нулевой уровень')
            ax2.set_title('Сезонная составляющая')
            ax2.set_xlabel('Временной индекс')
            ax2.set_ylabel('Значение')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Сохраняем в отдельный файл
            filename = f"{output_dir}/decomposition_{percentage}percent_chebyshev.png"
            plt.savefig(filename, dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
            
            print(f"    График декомпозиции Чебышева {percentage}% сохранен: {filename}")
            
        else:
            print(f"    Ошибка декомпозиции Чебышева для {percentage}% данных")

def create_individual_prophet_plots(series: pd.Series, output_dir: str):
    """Создать отдельные графики декомпозиции Prophet для каждого процента данных"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Вычисляем длины интервалов в процентах от общего количества данных
    total_length = len(series)
    interval_lengths = [int(total_length * p / 100) for p in DECOMPOSITION_PERCENTAGES]
    
    for length, percentage in zip(interval_lengths, DECOMPOSITION_PERCENTAGES):
        # Берем данные нужной длины - ТОЧНО ТАКИЕ ЖЕ КАК ДЛЯ ЧЕБЫШЕВА
        if len(series) > length:
            test_series = series.iloc[:length]
        else:
            test_series = series
        
        print(f"    Prophet {percentage}%: данные от {test_series.index[0]} до {test_series.index[-1]}")
        print(f"    Prophet {percentage}%: значения от {test_series.values[0]:.2f} до {test_series.values[-1]:.2f}")
        
        # Выполняем декомпозицию Prophet
        prophet_results = build_prophet_model(test_series)
        
        if 'error' not in prophet_results:
            # Используем те же индексы, что и для Чебышева для согласованности
            x = range(len(test_series))
            
            # Создаем файл с 2 подграфиками для каждого процента
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            fig.suptitle(f'Декомпозиция временного ряда - {percentage}% данных (n={length})\nProphet', 
                        fontsize=14, fontweight='bold')
            
            # График 1: Тренд + Исходные данные
            ax1.plot(x, test_series.values, 'b-', linewidth=2, label='Исходные данные', alpha=0.8)
            ax1.plot(x, prophet_results['trend'], 'r--', linewidth=2, label='Тренд', alpha=0.8)
            ax1.set_title(f'Тренд и исходные данные (RMSE: {prophet_results["rmse"]:.4f}, R²: {prophet_results["r2_score"]:.4f})')
            ax1.set_xlabel('Временной индекс')
            ax1.set_ylabel('Значение')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # График 2: Сезонность
            ax2.plot(x, prophet_results['seasonal'], 'g-', linewidth=2, label='Сезонная компонента', alpha=0.8)
            ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3, label='Нулевой уровень')
            ax2.set_title('Сезонная составляющая')
            ax2.set_xlabel('Временной индекс')
            ax2.set_ylabel('Значение')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Сохраняем в отдельный файл
            filename = f"{output_dir}/decomposition_{percentage}percent_prophet.png"
            plt.savefig(filename, dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
            
            print(f"    График декомпозиции Prophet {percentage}% сохранен: {filename}")
            
        else:
            print(f"    Ошибка декомпозиции Prophet для {percentage}% данных")

def create_comparison_table(series: pd.Series, output_dir: str):
    """Создать таблицу 3 с результатами декомпозиции"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Вычисляем длины интервалов в процентах от общего количества данных
    total_length = len(series)
    interval_lengths = [int(total_length * p / 100) for p in DECOMPOSITION_PERCENTAGES]
    
    # Подготовка данных для таблицы
    table_data_chebyshev = []
    table_data_prophet = []
    
    for length, percentage in zip(interval_lengths, DECOMPOSITION_PERCENTAGES):
        # Берем данные нужной длины
        if len(series) > length:
            test_series = series.iloc[:length]
        else:
            test_series = series
        
        # Декомпозиция Чебышева
        chebyshev_results = fourier_decomposition(test_series)
        if 'error' not in chebyshev_results:
            table_data_chebyshev.append({
                'Длина мерного интервала': length,
                'Лучшая модель': 'Многочлены Чебышева',
                'Среднеквадратичное отклонение': f"{chebyshev_results['rmse']:.4f}",
            })
        
        # Декомпозиция Prophet
        prophet_results = build_prophet_model(test_series)
        if 'error' not in prophet_results:
            # Определяем период на основе длины интервала
            if length <= 70:
                period = 30
            elif length <= 150:
                period = 90
            elif length <= 200:
                period = 7
            elif length <= 240:
                period = 60
            else:
                period = 120
            
            table_data_prophet.append({
                'Длина мерного интервала': length,
                'Период': period,
                'Среднеквадратичное отклонение': f"{prophet_results['rmse']:.4f}",
            })
    
    # Создаем DataFrame
    chebyshev_df = pd.DataFrame(table_data_chebyshev)
    prophet_df = pd.DataFrame(table_data_prophet)
    
    # Сохраняем в Excel
    filename = f"{output_dir}/table_3_decomposition_results.xlsx"
    
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        chebyshev_df.to_excel(writer, index=False, sheet_name="3_1_Чебышев")
        prophet_df.to_excel(writer, index=False, sheet_name="3_2_Prophet")
    
    print(f"Таблица 3 сохранена: {filename}")
    
    # Выводим таблицы в консоль
    print("\n" + "="*60)
    print("3.1 ТАБЛИЦА ДЕКОМПОЗИЦИИ (Многочлены Чебышева)")
    print("="*60)
    if not chebyshev_df.empty:
        print(chebyshev_df.to_string(index=False))
    else:
        print("Нет данных для таблицы Чебышева")
    
    print("\n" + "="*60)
    print("3.2 ТАБЛИЦА ДЕКОМпозиции (Prophet)")
    print("="*60)
    if not prophet_df.empty:
        print(prophet_df.to_string(index=False))
    else:
        print("Нет данных для таблицы Prophet")
    
    return chebyshev_df, prophet_df

def create_detailed_component_tables(series: pd.Series, output_dir: str):
    """Создать детальные таблицы компонентов для обеих методов"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Берем 90% данных для детального анализа
    total_length = len(series)
    length = int(total_length * 0.9)
    test_series = series.iloc[:length]
    
    # Таблица для Чебышева
    chebyshev_results = fourier_decomposition(test_series)
    if 'error' not in chebyshev_results:
        chebyshev_table = {
            'Компонента': ['Исходные данные', 'Тренд', 'Сезонность', 'Остатки', 'Восстановленный ряд'],
            'Среднее значение': [
                np.mean(test_series.values),
                np.mean(chebyshev_results['trend']),
                np.mean(chebyshev_results['seasonal']),
                np.mean(chebyshev_results['residuals']),
                np.mean(chebyshev_results['reconstructed'])
            ],
            'Стандартное отклонение': [
                np.std(test_series.values),
                np.std(chebyshev_results['trend']),
                np.std(chebyshev_results['seasonal']),
                np.std(chebyshev_results['residuals']),
                np.std(chebyshev_results['reconstructed'])
            ],
            'Минимальное значение': [
                np.min(test_series.values),
                np.min(chebyshev_results['trend']),
                np.min(chebyshev_results['seasonal']),
                np.min(chebyshev_results['residuals']),
                np.min(chebyshev_results['reconstructed'])
            ],
            'Максимальное значение': [
                np.max(test_series.values),
                np.max(chebyshev_results['trend']),
                np.max(chebyshev_results['seasonal']),
                np.max(chebyshev_results['residuals']),
                np.max(chebyshev_results['reconstructed'])
            ]
        }
        chebyshev_df = pd.DataFrame(chebyshev_table)
    else:
        chebyshev_df = pd.DataFrame()
    
    # Таблица для Prophet
    prophet_results = build_prophet_model(test_series)
    if 'error' not in prophet_results:
        prophet_table = {
            'Компонента': ['Исходные данные', 'Тренд', 'Сезонность', 'Остатки', 'Восстановленный ряд'],
            'Среднее значение': [
                np.mean(test_series.values),
                np.mean(prophet_results['trend']),
                np.mean(prophet_results['seasonal']),
                np.mean(prophet_results['residuals']),
                np.mean(prophet_results['reconstructed'])
            ],
            'Стандартное отклонение': [
                np.std(test_series.values),
                np.std(prophet_results['trend']),
                np.std(prophet_results['seasonal']),
                np.std(prophet_results['residuals']),
                np.std(prophet_results['reconstructed'])
            ],
            'Минимальное значение': [
                np.min(test_series.values),
                np.min(prophet_results['trend']),
                np.min(prophet_results['seasonal']),
                np.min(prophet_results['residuals']),
                np.min(prophet_results['reconstructed'])
            ],
            'Максимальное значение': [
                np.max(test_series.values),
                np.max(prophet_results['trend']),
                np.max(prophet_results['seasonal']),
                np.max(prophet_results['residuals']),
                np.max(prophet_results['reconstructed'])
            ]
        }
        prophet_df = pd.DataFrame(prophet_table)
    else:
        prophet_df = pd.DataFrame()
    
    # Сохраняем детальные таблицы
    filename = f"{output_dir}/detailed_components_analysis.xlsx"
    
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        if not chebyshev_df.empty:
            chebyshev_df.to_excel(writer, index=False, sheet_name="Чебышев_компоненты")
        if not prophet_df.empty:
            prophet_df.to_excel(writer, index=False, sheet_name="Prophet_компоненты")
    
    print(f"Детальные таблицы компонентов сохранены: {filename}")
    
    return chebyshev_df, prophet_df

def main():
    """Основная функция"""
    print("=== Лабораторная работа №2 (графики и таблицы) ===")
    print("Создание графиков декомпозиции и таблиц для отчета\n")
    
    # Проверяем входной файл
    if not os.path.exists(INPUT_FILE):
        print(f"Файл {INPUT_FILE} не найден. Запустите сначала лабораторную №1.")
        return
    
    # Загружаем данные
    try:
        series = read_excel_series(INPUT_FILE, INPUT_SHEET, DATE_COLUMN, VALUE_COLUMN)
        print(f"Загружен ряд: {len(series)} точек")
        print(f"Период: {series.index.min()} до {series.index.max()}")
        print(f"Диапазон значений: от {series.values.min():.2f} до {series.values.max():.2f}")
        
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return
    
    # Создаем выходную директорию
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\nСоздание графиков и таблиц...")
    print("=" * 60)
    
    # 1. Индивидуальные графики декомпозиции Чебышева
    print("Создание индивидуальных графиков декомпозиции (многочлены Чебышева)...")
    create_individual_chebyshev_plots(series, OUTPUT_DIR)
    
    # 2. Индивидуальные графики декомпозиции Prophet
    print("Создание индивидуальных графиков декомпозиции (Prophet)...")
    create_individual_prophet_plots(series, OUTPUT_DIR)
    
    # 3. Таблица 3 с результатами
    print("Создание таблицы 3 с результатами декомпозиции...")
    chebyshev_df, prophet_df = create_comparison_table(series, OUTPUT_DIR)
    
    # 4. Детальные таблицы компонентов
    print("Создание детальных таблиц компонентов...")
    create_detailed_component_tables(series, OUTPUT_DIR)
    
    # Итоговый отчет
    print(f"\n" + "=" * 60)
    print("ВСЕ ФАЙЛЫ СОЗДАНЫ:")
    print(f"Графики Чебышева:")
    for percentage in DECOMPOSITION_PERCENTAGES:
        print(f"  - {OUTPUT_DIR}/decomposition_{percentage}percent_chebyshev.png")
    print(f"Графики Prophet:")
    for percentage in DECOMPOSITION_PERCENTAGES:
        print(f"  - {OUTPUT_DIR}/decomposition_{percentage}percent_prophet.png")
    print(f"Таблицы:")
    print(f"  - {OUTPUT_DIR}/table_3_decomposition_results.xlsx")
    print(f"  - {OUTPUT_DIR}/detailed_components_analysis.xlsx")
    
    # Сводка по результатам
    if not chebyshev_df.empty and not prophet_df.empty:
        print(f"\nСВОДКА РЕЗУЛЬТАТОВ:")
        print(f"Многочлены Чебышева - Лучший RMSE: {chebyshev_df['Среднеквадратичное отклонение'].min()}")
        print(f"Prophet - Лучший RMSE: {prophet_df['Среднеквадратичное отклонение'].min()}")
    
    print(f"\nРабота завершена!")

if __name__ == "__main__":
    main()