#!/usr/bin/env python3
"""
Лабораторная работа №2 - РАСШИРЕННАЯ ВЕРСИЯ
Построение регрессионных, авторегрессионных моделей и моделей в пространстве состояний

НАСТРОЙКА ПАРАМЕТРОВ:
Все основные параметры вынесены в раздел КОНФИГУРАЦИОННЫЕ ПАРАМЕТРЫ (строки 20-63).
Для изменения поведения программы просто измените соответствующие константы.
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
from scipy.signal import periodogram

# ===================================================================
# КОНФИГУРАЦИОННЫЕ ПАРАМЕТРЫ - НАСТРОЙКИ ЛАБОРАТОРНОЙ РАБОТЫ
# ===================================================================

# Пути к файлам
INPUT_FILE = "out/clean.xlsx"
OUTPUT_DIR = "out/lab2"
INPUT_SHEET = "clean"
DATE_COLUMN = "data"
VALUE_COLUMN = "curs"

# Параметры ARIMA моделей
MAX_P = 5                    # Максимальный порядок авторегрессии
MAX_Q = 5                    # Максимальный порядок скользящего среднего
ARIMA_TOP_MODELS = 3         # Количество лучших моделей для вывода

# Параметры декомпозиции
MAX_POLY_DEGREE = 3          # Максимальная степень полиномиального тренда
N_HARMONICS = 5              # Количество гармоник для Фурье-анализа

# Параметры BSTS моделей
BSTS_MAX_ITER = 100         # Максимальное количество итераций

# Параметры Prophet
PROPHET_DAILY_SEASONALITY = False
PROPHET_WEEKLY_SEASONALITY = True
PROPHET_YEARLY_SEASONALITY = True
PROPHET_SEASONALITY_MODE = 'additive'

# Тестируемые длины интервалов
INTERVAL_LENGTHS = [50, 100, 200, 300]  # Последнее значение будет ограничено длиной ряда

# Параметры оценки качества
TEST_SIZE = 0.2             # Доля тестовой выборки для оценки качества
CONFIDENCE_LEVEL = 1.96     # Коэффициент для 95% доверительного интервала
MIN_TEST_SIZE = 5           # Минимальный размер тестовой выборки

# Параметры графиков
FIGURE_DPI = 150            # Разрешение сохраняемых графиков
FIGURE_SIZE_MAIN = (15, 8)  # Размер основных графиков
FIGURE_SIZE_COMPARISON = (20, 15)  # Размер графиков сравнения
FIGURE_SIZE_CI = (15, 10)   # Размер графиков доверительных интервалов

# ===================================================================

# Полностью отключаем все предупреждения
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

# Дополнительное подавление специфичных предупреждений statsmodels
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning, ConvergenceWarning
warnings.filterwarnings("ignore", category=ValueWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Импорты для моделей временных рядов
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.structural import UnobservedComponents
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

def check_stationarity(series: pd.Series) -> dict:
    """Детальная проверка стационарности"""
    results = {}
    
    try:
        # ADF тест
        adf_result = adfuller(series.dropna(), maxlag=None, autolag='AIC')
        results['adf'] = {
            'statistic': adf_result[0],
            'p_value': adf_result[1],
            'critical_values': adf_result[4],
            'is_stationary': adf_result[1] < 0.05
        }
        
        # KPSS тест
        kpss_result = kpss(series.dropna(), regression='c', nlags="auto")
        results['kpss'] = {
            'statistic': kpss_result[0],
            'p_value': kpss_result[1],
            'critical_values': kpss_result[3],
            'is_stationary': kpss_result[1] > 0.05
        }
        
        # Итоговое решение (ряд стационарен если оба теста согласны)
        results['is_stationary'] = (results['adf']['is_stationary'] and 
                                  results['kpss']['is_stationary'])
        
    except Exception as e:
        print(f"Ошибка в тестах стационарности: {e}")
        results = {'is_stationary': False, 'error': str(e)}
    
    return results

def find_best_arima_extended(series: pd.Series, max_p: int = MAX_P, max_q: int = MAX_Q) -> tuple:
    """Расширенный поиск лучших параметров ARIMA с автоматическим определением d"""
    best_aic = float('inf')
    best_params = (1, 1, 1)
    best_model = None
    
    # Определяем d через проверку стационарности
    stationarity = check_stationarity(series)
    d = 0 if stationarity.get('is_stationary', False) else 1
    
    # Если ряд нестационарен даже после первой разности, пробуем d=2
    if d == 1:
        diff_series = series.diff().dropna()
        if len(diff_series) > 10:
            diff_stationarity = check_stationarity(diff_series)
            if not diff_stationarity.get('is_stationary', False):
                d = 2
    
    print(f"    Определен порядок интегрирования d = {d}")
    
    # Поиск по сетке параметров
    search_results = []
    
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ARIMA(series, order=(p, d, q)).fit()
                    
                    search_results.append({
                        'p': p, 'd': d, 'q': q,
                        'aic': model.aic,
                        'bic': model.bic,
                        'model': model
                    })
                    
                    if model.aic < best_aic:
                        best_aic = model.aic
                        best_params = (p, d, q)
                        best_model = model
                            
            except Exception as e:
                continue
    
    # Сортируем результаты по AIC
    search_results.sort(key=lambda x: x['aic'])
    
    return best_params, best_model, search_results[:ARIMA_TOP_MODELS], d
def build_enhanced_bsts(series: pd.Series) -> tuple:
    """Улучшенная BSTS модель с автоматическим выбором компонентов"""
    best_model = None
    best_aic = float('inf')
    best_config = None
    
    # Тестируемые конфигурации
    configs = [
        {'level': 'local level', 'seasonal': None, 'trend': False},
        {'level': 'local linear trend', 'seasonal': None, 'trend': False},
        {'level': 'local level', 'seasonal': 12, 'trend': False},
        {'level': 'local linear trend', 'seasonal': 12, 'trend': False},
        {'level': 'random walk', 'seasonal': None, 'trend': False},
        {'level': 'random walk', 'seasonal': 12, 'trend': False},
    ]
    
    for config in configs:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = UnobservedComponents(
                    series,
                    level=config['level'],
                    seasonal=config['seasonal'],
                    trend=config['trend']
                )
                fitted = model.fit(maxiter=BSTS_MAX_ITER)
                
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_model = fitted
                    best_config = config
                    
        except Exception as e:
            continue
    
    if best_model:
        print(f"    Лучшая BSTS конфигурация: {best_config}")
    
    return best_model, best_aic

def fourier_decomposition(series: pd.Series, n_harmonics: int = N_HARMONICS) -> dict:
    """Декомпозиция с использованием Фурье-анализа"""
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
        
        # Проверка нормальности остатков
        try:
            jb_result = jarque_bera(residuals)
            jb_stat = jb_result[0]
            jb_pvalue = jb_result[1]
        except:
            jb_stat, jb_pvalue = 0, 1
            
        try:
            shapiro_stat, shapiro_pvalue = stats.shapiro(residuals[:min(5000, len(residuals))])
        except:
            shapiro_stat, shapiro_pvalue = 0, 1
        
        normality_test = {
            'jarque_bera': {'statistic': jb_stat, 'p_value': jb_pvalue, 'is_normal': jb_pvalue > 0.05},
            'shapiro': {'statistic': shapiro_stat, 'p_value': shapiro_pvalue, 'is_normal': shapiro_pvalue > 0.05}
        }
        
        # Метрики качества
        reconstructed = best_trend + seasonal
        rmse = np.sqrt(mean_squared_error(values, reconstructed))
        mae = mean_absolute_error(values, reconstructed)
        r2_total = r2_score(values, reconstructed)
        
        # Доверительный интервал
        residual_std = np.std(residuals)
        confidence_95 = CONFIDENCE_LEVEL * residual_std
        
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
            'confidence_95': confidence_95,
            'normality_test': normality_test
        }
        
    except Exception as e:
        print(f"    Ошибка в декомпозиции: {e}")
        return {'error': str(e), 'rmse': float('inf'), 'confidence_95': float('inf')}

def build_prophet_model(series: pd.Series) -> dict:
    """Построение модели Prophet"""
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
        
        # Прогноз на тех же данных для оценки качества
        forecast = model.predict(df)
        
        # Метрики качества
        rmse = np.sqrt(mean_squared_error(df['y'], forecast['yhat']))
        mae = mean_absolute_error(df['y'], forecast['yhat'])
        r2 = r2_score(df['y'], forecast['yhat'])
        
        # Доверительный интервал (средняя ширина)
        confidence_95 = np.mean(forecast['yhat_upper'] - forecast['yhat_lower']) / 2
        
        return {
            'model': model,
            'forecast': forecast,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'confidence_95': confidence_95
        }
        
    except Exception as e:
        print(f"    Ошибка в Prophet: {e}")
        return {'error': str(e), 'rmse': float('inf'), 'confidence_95': float('inf')}

def evaluate_forecast_enhanced(model, series: pd.Series, test_size: float = TEST_SIZE, model_type: str = 'arima') -> dict:
    """Улучшенная оценка качества прогноза"""
    if model is None:
        return {'rmse': float('inf'), 'mae': float('inf'), 'confidence_95': float('inf')}
    
    try:
        split_idx = int(len(series) * (1 - test_size))
        if split_idx >= len(series) - MIN_TEST_SIZE:
            split_idx = len(series) - MIN_TEST_SIZE
        
        train_series = series.iloc[:split_idx]
        test_series = series.iloc[split_idx:]
        
        if len(test_series) == 0:
            test_series = series.iloc[-MIN_TEST_SIZE:]
        
        # Прогноз в зависимости от типа модели
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if model_type == 'arima':
                try:
                    forecast_result = model.forecast(steps=len(test_series), alpha=0.05)
                    if hasattr(forecast_result, 'predicted_mean'):
                        forecast = forecast_result.predicted_mean
                        conf_int = forecast_result.conf_int()
                        confidence_95 = np.mean(conf_int.iloc[:, 1] - conf_int.iloc[:, 0]) / 2
                    else:
                        forecast = forecast_result
                        confidence_95 = CONFIDENCE_LEVEL * np.std(test_series.values - forecast) if len(forecast) == len(test_series) else CONFIDENCE_LEVEL * np.std(test_series.values)
                except:
                    forecast = np.array([test_series.mean()] * len(test_series))
                    confidence_95 = CONFIDENCE_LEVEL * np.std(test_series.values)
                    
            elif model_type == 'bsts':
                try:
                    forecast_result = model.forecast(steps=len(test_series))
                    if hasattr(forecast_result, 'predicted_mean'):
                        forecast = forecast_result.predicted_mean
                    else:
                        forecast = forecast_result
                    confidence_95 = CONFIDENCE_LEVEL * np.std(test_series.values - forecast) if len(forecast) == len(test_series) else CONFIDENCE_LEVEL * np.std(test_series.values)
                except:
                    forecast = np.array([test_series.mean()] * len(test_series))
                    confidence_95 = CONFIDENCE_LEVEL * np.std(test_series.values)
                
            else:  # other models
                try:
                    forecast = model.forecast(steps=len(test_series))
                    confidence_95 = CONFIDENCE_LEVEL * np.std(test_series.values - forecast) if len(forecast) == len(test_series) else CONFIDENCE_LEVEL * np.std(test_series.values)
                except:
                    forecast = np.array([test_series.mean()] * len(test_series))
                    confidence_95 = CONFIDENCE_LEVEL * np.std(test_series.values)
        
        # Метрики
        if len(forecast) == len(test_series):
            rmse = np.sqrt(mean_squared_error(test_series.values, forecast))
            mae = mean_absolute_error(test_series.values, forecast)
            # Дополнительные метрики
            mape = np.mean(np.abs((test_series.values - forecast) / np.where(test_series.values != 0, test_series.values, 1))) * 100
        else:
            rmse = float('inf')
            mae = float('inf')
            mape = float('inf')
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'confidence_95': confidence_95
        }
        
    except Exception as e:
        return {'rmse': float('inf'), 'mae': float('inf'), 'mape': float('inf'), 'confidence_95': float('inf')}
def run_comprehensive_analysis(series: pd.Series, length: int) -> dict:
    """Комплексный анализ для конкретной длины интервала"""
    print(f"  Длина {length}...")
    
    # Берем данные нужной длины
    if len(series) > length:
        test_series = series.iloc[:length]
    else:
        test_series = series
    
    results = {'length': length}
    
    # 1. ARIMA
    print(f"    Построение ARIMA модели...")
    try:
        arima_model, arima_params, arima_aic = find_best_arima_extended(test_series)
        if arima_model:
            arima_metrics = evaluate_forecast_enhanced(arima_model, test_series, model_type='arima')
        
        results['arima'] = {
            'p': arima_params[0],
            'd': arima_params[1],
            'q': arima_params[2],
            'rmse': arima_metrics['rmse'],
                'mae': arima_metrics['mae'],
                'mape': arima_metrics.get('mape', 0),
            'confidence_95': arima_metrics['confidence_95'],
            'aic': arima_aic
        }
        print(f"    ARIMA({arima_params[0]},{arima_params[1]},{arima_params[2]}) - RMSE: {arima_metrics['rmse']:.4f}")
        else:
            results['arima'] = {'error': 'Не удалось построить модель'}
    except Exception as e:
        print(f"    ARIMA ошибка: {e}")
        results['arima'] = {'error': str(e)}
    
    # 2. BSTS
    print(f"    Построение BSTS модели...")
    try:
        bsts_model, bsts_aic = build_enhanced_bsts(test_series)
        if bsts_model:
            bsts_metrics = evaluate_forecast_enhanced(bsts_model, test_series, model_type='bsts')
            results['bsts'] = {
                'rmse': bsts_metrics['rmse'],
                'mae': bsts_metrics['mae'],
                'mape': bsts_metrics.get('mape', 0),
                'confidence_95': bsts_metrics['confidence_95'],
                'aic': bsts_aic
            }
            print(f"    BSTS - RMSE: {bsts_metrics['rmse']:.4f}")
        else:
            results['bsts'] = {'error': 'Не удалось построить модель'}
    except Exception as e:
        print(f"    BSTS ошибка: {e}")
        results['bsts'] = {'error': str(e)}
    
    # 3. Декомпозиция (Фурье + полиномиальная регрессия)
    print(f"    Выполнение декомпозиции...")
    try:
        decomp_results = fourier_decomposition(test_series)
        if 'error' not in decomp_results:
            results['decomposition'] = {
                'rmse': decomp_results['rmse'],
                'mae': decomp_results['mae'],
                'r2_score': decomp_results['r2_score'],
                'confidence_95': decomp_results['confidence_95'],
                'trend_degree': decomp_results['trend_degree'],
                'n_harmonics': decomp_results['n_harmonics'],
                'normality_test': decomp_results['normality_test']
            }
            print(f"    Декомпозиция - RMSE: {decomp_results['rmse']:.4f}, R2: {decomp_results['r2_score']:.4f}")
        else:
            results['decomposition'] = decomp_results
    except Exception as e:
        print(f"    Декомпозиция ошибка: {e}")
        results['decomposition'] = {'error': str(e)}
    
    # 4. Prophet
    print(f"    Построение Prophet модели...")
    try:
        prophet_results = build_prophet_model(test_series)
        if 'error' not in prophet_results:
            results['prophet'] = {
                'rmse': prophet_results['rmse'],
                'mae': prophet_results['mae'],
                'r2_score': prophet_results['r2_score'],
                'confidence_95': prophet_results['confidence_95']
            }
            print(f"    Prophet - RMSE: {prophet_results['rmse']:.4f}, R2: {prophet_results['r2_score']:.4f}")
        else:
            results['prophet'] = prophet_results
    except Exception as e:
        print(f"    Prophet ошибка: {e}")
        results['prophet'] = {'error': str(e)}
    
    return results

def create_enhanced_summary_table(all_results: list, output_dir: str):
    """Создать расширенную сводную таблицу"""
    os.makedirs(output_dir, exist_ok=True)
    
    table_data = []
    
    for result in all_results:
        length = result['length']
        
        # ARIMA
        if 'arima' in result and 'error' not in result['arima']:
            arima = result['arima']
            table_data.append({
                'Модель': 'ARIMA',
                'Длина интервала': length,
                'p': arima.get('p', '-'),
                'd': arima.get('d', '-'),
                'q': arima.get('q', '-'),
                'RMSE': f"{arima.get('rmse', 0):.4f}",
                'MAE': f"{arima.get('mae', 0):.4f}",
                'MAPE, %': f"{arima.get('mape', 0):.2f}",
                'Доверительный интервал 95%': f"±{arima.get('confidence_95', 0):.4f}",
                'AIC': f"{arima.get('aic', 0):.2f}",
                'R2': '-'
            })
        
        # BSTS
        if 'bsts' in result and 'error' not in result['bsts']:
            bsts = result['bsts']
            table_data.append({
                'Модель': 'BSTS',
                'Длина интервала': length,
                'p': '-', 'd': '-', 'q': '-',
                'RMSE': f"{bsts.get('rmse', 0):.4f}",
                'MAE': f"{bsts.get('mae', 0):.4f}",
                'MAPE, %': f"{bsts.get('mape', 0):.2f}",
                'Доверительный интервал 95%': f"±{bsts.get('confidence_95', 0):.4f}",
                'AIC': f"{bsts.get('aic', 0):.2f}",
                'R2': '-'
            })
        
        # Декомпозиция
        if 'decomposition' in result and 'error' not in result['decomposition']:
            decomp = result['decomposition']
            table_data.append({
                'Модель': 'Декомпозиция (Фурье)',
                'Длина интервала': length,
                'p': '-', 'd': '-', 'q': '-',
                'RMSE': f"{decomp.get('rmse', 0):.4f}",
                'MAE': f"{decomp.get('mae', 0):.4f}",
                'MAPE, %': '-',
                'Доверительный интервал 95%': f"±{decomp.get('confidence_95', 0):.4f}",
                'AIC': '-',
                'R2': f"{decomp.get('r2_score', 0):.4f}"
            })
        
        # Prophet
        if 'prophet' in result and 'error' not in result['prophet']:
            prophet_res = result['prophet']
            table_data.append({
                'Модель': 'Prophet',
                'Длина интервала': length,
                'p': '-', 'd': '-', 'q': '-',
                'RMSE': f"{prophet_res.get('rmse', 0):.4f}",
                'MAE': f"{prophet_res.get('mae', 0):.4f}",
                'MAPE, %': '-',
                'Доверительный интервал 95%': f"±{prophet_res.get('confidence_95', 0):.4f}",
                'AIC': '-',
                'R2': f"{prophet_res.get('r2_score', 0):.4f}"
            })
    
    # Сохраняем в Excel
    df = pd.DataFrame(table_data)
    filename = f"{output_dir}/summary_results.xlsx"
    
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
    
    print(f"\nТаблица результатов: {filename}")
    return df

def create_comprehensive_plots(series: pd.Series, all_results: list, output_dir: str):
    """Создать комплексные графики для всех моделей"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. График исходного ряда
    plt.figure(figsize=FIGURE_SIZE_MAIN)
    plt.plot(series.index, series.values, label='Временной ряд', alpha=0.8, linewidth=1.5)
    plt.title('Исходные данные временного ряда', fontsize=16, fontweight='bold')
    plt.xlabel('Дата', fontsize=12)
    plt.ylabel('Значение', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/original_series_clean_series.png", dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    # 2. Сравнение моделей
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE_COMPARISON)
    fig.suptitle('Сравнение точности моделей по длинам интервалов', fontsize=16, fontweight='bold')
    
    # Подготовка данных для графиков
    lengths = [r['length'] for r in all_results]
    models_data = {
        'ARIMA': {'rmse': [], 'ci': []},
        'BSTS': {'rmse': [], 'ci': []},
        'Декомпозиция': {'rmse': [], 'ci': []},
        'Prophet': {'rmse': [], 'ci': []}
    }
    
    for result in all_results:
        # ARIMA
        if 'arima' in result and 'error' not in result['arima']:
            models_data['ARIMA']['rmse'].append(result['arima']['rmse'])
            models_data['ARIMA']['ci'].append(result['arima']['confidence_95'])
        else:
            models_data['ARIMA']['rmse'].append(np.nan)
            models_data['ARIMA']['ci'].append(np.nan)
        
        # BSTS
        if 'bsts' in result and 'error' not in result['bsts']:
            models_data['BSTS']['rmse'].append(result['bsts']['rmse'])
            models_data['BSTS']['ci'].append(result['bsts']['confidence_95'])
        else:
            models_data['BSTS']['rmse'].append(np.nan)
            models_data['BSTS']['ci'].append(np.nan)
        
        # Декомпозиция
        if 'decomposition' in result and 'error' not in result['decomposition']:
            models_data['Декомпозиция']['rmse'].append(result['decomposition']['rmse'])
            models_data['Декомпозиция']['ci'].append(result['decomposition']['confidence_95'])
        else:
            models_data['Декомпозиция']['rmse'].append(np.nan)
            models_data['Декомпозиция']['ci'].append(np.nan)
        
        # Prophet
        if 'prophet' in result and 'error' not in result['prophet']:
            models_data['Prophet']['rmse'].append(result['prophet']['rmse'])
            models_data['Prophet']['ci'].append(result['prophet']['confidence_95'])
        else:
            models_data['Prophet']['rmse'].append(np.nan)
            models_data['Prophet']['ci'].append(np.nan)
    
    # График RMSE
    ax1 = axes[0, 0]
    for model_name, data in models_data.items():
        ax1.plot(lengths, data['rmse'], marker='o', label=model_name, linewidth=2, markersize=8)
    ax1.set_xlabel('Длина интервала')
    ax1.set_ylabel('RMSE')
    ax1.set_title('Среднеквадратическое отклонение (RMSE)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График доверительных интервалов
    ax2 = axes[0, 1]
    for model_name, data in models_data.items():
        ax2.plot(lengths, data['ci'], marker='s', label=model_name, linewidth=2, markersize=8)
    ax2.set_xlabel('Длина интервала')
    ax2.set_ylabel('Доверительный интервал 95%')
    ax2.set_title('Ширина доверительных интервалов')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Барплот лучших моделей по RMSE
    ax3 = axes[1, 0]
    best_rmse_by_length = []
    best_model_by_length = []
    for i, length in enumerate(lengths):
        min_rmse = float('inf')
        best_model = 'None'
        for model_name, data in models_data.items():
            if not np.isnan(data['rmse'][i]) and data['rmse'][i] < min_rmse:
                min_rmse = data['rmse'][i]
                best_model = model_name
        best_rmse_by_length.append(min_rmse if min_rmse != float('inf') else 0)
        best_model_by_length.append(best_model)
    
    colors = sns.color_palette("husl", len(set(best_model_by_length)))
    color_map = {model: colors[i] for i, model in enumerate(set(best_model_by_length))}
    bar_colors = [color_map[model] for model in best_model_by_length]
    
    bars = ax3.bar(range(len(lengths)), best_rmse_by_length, color=bar_colors)
    ax3.set_xlabel('Длина интервала')
    ax3.set_ylabel('Лучший RMSE')
    ax3.set_title('Лучшая модель по RMSE для каждой длины')
    ax3.set_xticks(range(len(lengths)))
    ax3.set_xticklabels(lengths)
    
    # Добавляем подписи на столбцы
    for i, (bar, model) in enumerate(zip(bars, best_model_by_length)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{model}\n{height:.3f}',
                ha='center', va='bottom', fontsize=8)
    
    # График доверительных интервалов последней модели декомпозиции
    ax4 = axes[1, 1]
    if all_results and 'decomposition' in all_results[-1] and 'error' not in all_results[-1]['decomposition']:
        last_result = all_results[-1]
        # Получаем данные последней декомпозиции для демонстрации
        length = last_result['length']
        test_series = series.iloc[:length] if len(series) > length else series
        
        # Быстрая декомпозиция для графика
        decomp_demo = fourier_decomposition(test_series)
        if 'error' not in decomp_demo:
            x_demo = range(len(test_series))
            ax4.plot(x_demo, test_series.values, label='Исходные данные', alpha=0.7)
            ax4.plot(x_demo, decomp_demo['reconstructed'], label='Модель', linewidth=2)
            ax4.fill_between(x_demo, 
                           decomp_demo['reconstructed'] - decomp_demo['confidence_95'],
                           decomp_demo['reconstructed'] + decomp_demo['confidence_95'],
                           alpha=0.3, label='95% ДИ')
            ax4.set_title(f'Декомпозиция с доверительными интервалами\n(длина {length})')
            ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'Декомпозиция недоступна', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Декомпозиция с доверительными интервалами')
    
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/models_comparison_clean_series.png", dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    # 3. График доверительных интервалов
    plt.figure(figsize=FIGURE_SIZE_CI)
    
    # Берем последний результат для демонстрации доверительных интервалов
    if all_results:
        last_result = all_results[-1]
        length = last_result['length']
        test_series = series.iloc[:length] if len(series) > length else series
        
        plt.subplot(2, 2, 1)
        # ARIMA доверительные интервалы
        if 'arima' in last_result and 'error' not in last_result['arima']:
            try:
                # Простая демонстрация доверительного интервала
                arima_ci = last_result['arima']['confidence_95']
                x = range(min(50, len(test_series)))
                y = test_series.values[:len(x)]
                plt.plot(x, y, 'b-', label='Данные', alpha=0.7)
                plt.fill_between(x, y - arima_ci, y + arima_ci, alpha=0.3, label=f'95% ДИ (±{arima_ci:.3f})')
                plt.title('ARIMA - Доверительные интервалы')
                plt.legend()
                plt.grid(True, alpha=0.3)
            except:
                plt.text(0.5, 0.5, 'ARIMA недоступна', ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.subplot(2, 2, 2)
        # BSTS доверительные интервалы
        if 'bsts' in last_result and 'error' not in last_result['bsts']:
            try:
                bsts_ci = last_result['bsts']['confidence_95']
                x = range(min(50, len(test_series)))
                y = test_series.values[:len(x)]
                plt.plot(x, y, 'g-', label='Данные', alpha=0.7)
                plt.fill_between(x, y - bsts_ci, y + bsts_ci, alpha=0.3, label=f'95% ДИ (±{bsts_ci:.3f})')
                plt.title('BSTS - Доверительные интервалы')
                plt.legend()
                plt.grid(True, alpha=0.3)
            except:
                plt.text(0.5, 0.5, 'BSTS недоступна', ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.subplot(2, 2, 3)
        # Декомпозиция доверительные интервалы
        if 'decomposition' in last_result and 'error' not in last_result['decomposition']:
            try:
                decomp_ci = last_result['decomposition']['confidence_95']
                x = range(min(50, len(test_series)))
                y = test_series.values[:len(x)]
                plt.plot(x, y, 'r-', label='Данные', alpha=0.7)
                plt.fill_between(x, y - decomp_ci, y + decomp_ci, alpha=0.3, label=f'95% ДИ (±{decomp_ci:.3f})')
                plt.title('Декомпозиция - Доверительные интервалы')
                plt.legend()
                plt.grid(True, alpha=0.3)
            except:
                plt.text(0.5, 0.5, 'Декомпозиция недоступна', ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.subplot(2, 2, 4)
        # Prophet доверительные интервалы
        if 'prophet' in last_result and 'error' not in last_result['prophet']:
            try:
                prophet_ci = last_result['prophet']['confidence_95']
                x = range(min(50, len(test_series)))
                y = test_series.values[:len(x)]
                plt.plot(x, y, 'm-', label='Данные', alpha=0.7)
                plt.fill_between(x, y - prophet_ci, y + prophet_ci, alpha=0.3, label=f'95% ДИ (±{prophet_ci:.3f})')
                plt.title('Prophet - Доверительные интервалы')
    plt.legend()
    plt.grid(True, alpha=0.3)
            except:
                plt.text(0.5, 0.5, 'Prophet недоступен', ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confidence_intervals_clean_series.png", dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Графики сохранены в: {output_dir}/")

def main():
    """Основная функция"""
    print("=== Лабораторная работа №2 (расширенная версия) ===")
    print("Построение регрессионных, авторегрессионных моделей и моделей в пространстве состояний\n")
    
    # Настройки (используем константы из конфигурации)
    input_file = INPUT_FILE
    output_dir = OUTPUT_DIR
    
    # Проверяем входной файл
    if not os.path.exists(input_file):
        print(f"Файл {input_file} не найден. Запустите сначала лабораторную №1.")
        return
    
    # Загружаем данные
    try:
        series = read_excel_series(input_file, INPUT_SHEET, DATE_COLUMN, VALUE_COLUMN)
        print(f"Загружен ряд: {len(series)} точек")
        print(f"Период: {series.index.min()} - {series.index.max()}")
        
        # Детальная проверка стационарности
        stationarity_results = check_stationarity(series)
        if 'error' not in stationarity_results:
            print(f"\nРезультаты тестов стационарности:")
            print(f"  ADF тест: статистика={stationarity_results['adf']['statistic']:.4f}, p-value={stationarity_results['adf']['p_value']:.4f}")
            print(f"  KPSS тест: статистика={stationarity_results['kpss']['statistic']:.4f}, p-value={stationarity_results['kpss']['p_value']:.4f}")
            print(f"  Ряд стационарен: {stationarity_results['is_stationary']}")
        else:
            print(f"Ошибка в тестах стационарности: {stationarity_results['error']}")
        
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return
    
    # Длины интервалов для тестирования (используем константы)
    interval_lengths = [l for l in INTERVAL_LENGTHS if l <= len(series)]
    # Добавляем полную длину ряда, если она не входит в список
    if len(series) not in interval_lengths:
        interval_lengths.append(len(series))
    
    print(f"\nТестируемые интервалы: {interval_lengths}")
    print("=" * 60)
    
    # Запускаем комплексный анализ
    all_results = []
    
    for length in interval_lengths:
        print(f"\nАнализ интервала {length}:")
        print("-" * 40)
        result = run_comprehensive_analysis(series, length)
        all_results.append(result)
    
    # Создаем результаты
    print("\n" + "=" * 60)
    print("Создание отчетов и графиков...")
    
    results_df = create_enhanced_summary_table(all_results, output_dir)
    create_comprehensive_plots(series, all_results, output_dir)
    
    # Анализ лучших результатов
    print("\n" + "=" * 60)
    print("=== АНАЛИЗ ЛУЧШИХ РЕЗУЛЬТАТОВ ===")
    
    if not results_df.empty:
        print("\nПо каждой модели:")
        print("-" * 30)
        
        for model_name in results_df['Модель'].unique():
            model_data = results_df[results_df['Модель'] == model_name]
            if not model_data.empty and not model_data['RMSE'].str.contains('inf').any():
                best_idx = model_data['RMSE'].astype(float).idxmin()
                best_row = model_data.loc[best_idx]
                
                print(f"\n{model_name}:")
                print(f"  Лучшая длина интервала: {best_row['Длина интервала']}")
                print(f"  RMSE: {best_row['RMSE']}")
                print(f"  MAE: {best_row['MAE']}")
                if best_row['MAPE, %'] != '-':
                    print(f"  MAPE: {best_row['MAPE, %']}%")
                print(f"  Доверительный интервал: {best_row['Доверительный интервал 95%']}")
                if best_row['R2'] != '-':
                    print(f"  R2: {best_row['R2']}")
                if best_row['p'] != '-':
                    print(f"  Параметры: p={best_row['p']}, d={best_row['d']}, q={best_row['q']}")
    
        # Общий лучший результат
        numeric_rmse = results_df[~results_df['RMSE'].str.contains('inf')]['RMSE'].astype(float)
        if not numeric_rmse.empty:
            best_overall_idx = numeric_rmse.idxmin()
            best_overall = results_df.loc[best_overall_idx]
            
            print(f"\n" + "=" * 40)
            print("ЛУЧШИЙ РЕЗУЛЬТАТ ОБЩИЙ:")
            print(f"  Модель: {best_overall['Модель']}")
            print(f"  Длина интервала: {best_overall['Длина интервала']}")
            print(f"  RMSE: {best_overall['RMSE']}")
            print(f"  Доверительный интервал: {best_overall['Доверительный интервал 95%']}")
    
    print(f"\n" + "=" * 60)
    print(f"Все результаты сохранены в: {output_dir}/")
    print("Файлы:")
    print(f"  - summary_results.xlsx (детальная таблица)")
    print(f"  - original_series_clean_series.png (исходный ряд)")
    print(f"  - models_comparison_clean_series.png (сравнение моделей)")
    print(f"  - confidence_intervals_clean_series.png (доверительные интервалы)")
    print("\nЛабораторная работа №2 завершена!")

if __name__ == "__main__":
    main()
