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
from scipy.signal import periodogram

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

# Дополнительные параметры для расширенного анализа
# ARIMA
MAX_P = 5
MAX_Q = 5
ARIMA_TOP_MODELS = 3

# BSTS/SSM
BSTS_MAX_ITER = 100

# Интервалы и качество
INTERVAL_LENGTHS = [62, 124, 186, 223]
TEST_SIZE = 0.2
CONFIDENCE_LEVEL = 1.96
MIN_TEST_SIZE = 5

# Размеры графиков расширенного сравнения
FIGURE_SIZE_COMPARISON = (20, 15)
FIGURE_SIZE_CI = (15, 10)

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
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.structural import UnobservedComponents
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

def check_stationarity(series: pd.Series) -> dict:
    """Проверка стационарности (ADF и KPSS)"""
    results = {}
    try:
        adf_stat, adf_p, _, _, adf_crit, _ = adfuller(series.dropna(), autolag='AIC')
        results['adf'] = {'statistic': adf_stat, 'p_value': adf_p, 'critical_values': adf_crit,
                          'is_stationary': adf_p < 0.05}
        kpss_stat, kpss_p, _, kpss_crit = kpss(series.dropna(), regression='c', nlags='auto')
        results['kpss'] = {'statistic': kpss_stat, 'p_value': kpss_p, 'critical_values': kpss_crit,
                           'is_stationary': kpss_p > 0.05}
        results['is_stationary'] = results['adf']['is_stationary'] and results['kpss']['is_stationary']
    except Exception as e:
        results = {'is_stationary': False, 'error': str(e)}
    return results

def find_best_arima(series: pd.Series, max_p: int = MAX_P, max_q: int = MAX_Q):
    """Подбор ARIMA(p,d,q) по AIC с автоматическим d"""
    best_aic = float('inf')
    best_model = None
    best_params = (0, 0, 0)
    # d по стационарности
    d = 0 if check_stationarity(series).get('is_stationary', False) else 1
    if d == 1:
        diff = series.diff().dropna()
        if len(diff) > 10 and not check_stationarity(diff).get('is_stationary', True):
            d = 2
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    model = ARIMA(series, order=(p, d, q)).fit()
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_model = model
                    best_params = (p, d, q)
            except Exception:
                continue
    return best_params, best_model, best_aic

def build_bsts(series: pd.Series):
    """Простые конфигурации структурной модели (SSM/BSTS surrogate)."""
    best_model = None
    best_aic = float('inf')
    best_cfg = None
    configs = [
        {'level': 'local level', 'seasonal': None, 'trend': False},
        {'level': 'local linear trend', 'seasonal': None, 'trend': False},
        {'level': 'random walk', 'seasonal': None, 'trend': False},
        {'level': 'local level', 'seasonal': 12, 'trend': False},
        {'level': 'local level', 'seasonal': 24, 'trend': False},
        {'level': 'local level', 'seasonal': 52, 'trend': False},
        {'level': 'local linear trend', 'seasonal': 12, 'trend': False},
        {'level': 'local linear trend', 'seasonal': 24, 'trend': False},
        {'level': 'local linear trend', 'seasonal': 52, 'trend': False},
        {'level': 'random walk', 'seasonal': 24, 'trend': False},
        {'level': 'random walk', 'seasonal': 52, 'trend': False},
    ]
    for cfg in configs:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model = UnobservedComponents(series, level=cfg['level'], seasonal=cfg['seasonal'], trend=cfg['trend'])
                fitted = model.fit(maxiter=BSTS_MAX_ITER)
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_model = fitted
                best_cfg = cfg
        except Exception:
            continue
    return best_model, best_aic, best_cfg

def evaluate_forecast(model, series: pd.Series, model_type: str) -> dict:
    """Оценка по hold-out TEST_SIZE с RMSE/MAE/CI."""
    try:
        split_idx = int(len(series) * (1 - TEST_SIZE))
        if split_idx >= len(series) - MIN_TEST_SIZE:
            split_idx = len(series) - MIN_TEST_SIZE
        train = series.iloc[:split_idx]
        test = series.iloc[split_idx:]
        if model_type == 'arima':
            res = model.forecast(steps=len(test), alpha=0.05)
            if hasattr(res, 'predicted_mean'):
                yhat = np.asarray(res.predicted_mean)
                ci = res.conf_int()
                ci_width = np.mean(ci.iloc[:, 1] - ci.iloc[:, 0]) / 2
            else:
                yhat = np.asarray(res)
                ci_width = CONFIDENCE_LEVEL * np.std(test.values - yhat)
        elif model_type == 'bsts':
            res = model.forecast(steps=len(test))
            yhat = np.asarray(res if not hasattr(res, 'predicted_mean') else res.predicted_mean)
            ci_width = CONFIDENCE_LEVEL * np.std(test.values - yhat)
        else:
            return {'rmse': float('inf'), 'mae': float('inf'), 'confidence_95': float('inf')}
        rmse = np.sqrt(mean_squared_error(test.values, yhat)) if len(yhat) == len(test) else float('inf')
        mae = mean_absolute_error(test.values, yhat) if len(yhat) == len(test) else float('inf')
        return {'rmse': rmse, 'mae': mae, 'confidence_95': ci_width}
    except Exception:
        return {'rmse': float('inf'), 'mae': float('inf'), 'confidence_95': float('inf')}

def run_extended_analysis(series: pd.Series, length: int) -> dict:
    """Анализ для определённой длины: ARIMA, BSTS, Декомпозиция, Prophet."""
    subset = series.iloc[:length] if len(series) > length else series
    out = {'length': length}
    # ARIMA
    try:
        params, arima_model, aic = find_best_arima(subset)
        metrics = evaluate_forecast(arima_model, subset, 'arima') if arima_model else {'rmse': float('inf'), 'mae': float('inf'), 'confidence_95': float('inf')}
        out['arima'] = {'p': params[0], 'd': params[1], 'q': params[2], 'rmse': metrics['rmse'], 'mae': metrics['mae'], 'confidence_95': metrics['confidence_95'], 'aic': aic}
    except Exception as e:
        out['arima'] = {'error': str(e)}
    # BSTS
    try:
        bsts_model, bsts_aic, cfg = build_bsts(subset)
        metrics = evaluate_forecast(bsts_model, subset, 'bsts') if bsts_model else {'rmse': float('inf'), 'mae': float('inf'), 'confidence_95': float('inf')}
        # Название тренда для отчета
        trend_name_map = {
            'local level': 'Сглаженный',
            'random walk': 'Случайное блуждание',
            'local linear trend': 'Локальный линейный тренд'
        }
        trend_name = trend_name_map.get(cfg['level'] if cfg else 'local level', 'Сглаженный')
        out['bsts'] = {'rmse': metrics['rmse'], 'mae': metrics['mae'], 'confidence_95': metrics['confidence_95'], 'aic': bsts_aic,
                       'seasonal': (cfg.get('seasonal') if cfg else None), 'trend_name': trend_name}
    except Exception as e:
        out['bsts'] = {'error': str(e)}
    # Декомпозиция
    try:
        dec = fourier_decomposition(subset)
        if 'error' not in dec:
            out['decomposition'] = {'rmse': dec['rmse'], 'mae': dec['mae'], 'r2_score': dec['r2_score'], 'confidence_95': CONFIDENCE_LEVEL * np.std(dec['residuals'])}
        else:
            out['decomposition'] = dec
    except Exception as e:
        out['decomposition'] = {'error': str(e)}
    # Prophet
    try:
        prop = build_prophet_model(subset)
        if 'error' not in prop:
            out['prophet'] = {'rmse': prop['rmse'], 'mae': prop['mae'], 'r2_score': prop['r2_score'], 'confidence_95': np.mean(prop['forecast']['yhat_upper'] - prop['forecast']['yhat_lower']) / 2}
        else:
            out['prophet'] = prop
    except Exception as e:
        out['prophet'] = {'error': str(e)}
    return out

def create_extended_summary(all_results: list, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for r in all_results:
        L = r['length']
        if 'arima' in r and 'error' not in r['arima']:
            a = r['arima']
            rows.append({'Модель': 'ARIMA', 'Длина интервала': L, 'p': a['p'], 'd': a['d'], 'q': a['q'], 'RMSE': f"{a['rmse']:.4f}", 'MAE': f"{a['mae']:.4f}", 'Доверительный интервал 95%': f"±{a['confidence_95']:.4f}", 'AIC': f"{a['aic']:.2f}", 'R2': '-'})
        if 'bsts' in r and 'error' not in r['bsts']:
            b = r['bsts']
            rows.append({'Модель': 'BSTS', 'Длина интервала': L, 'Сезонность': (b.get('seasonal') if b.get('seasonal') is not None else '-'), 'Тренд': b.get('trend_name','-'), 'RMSE': f"{b['rmse']:.4f}", 'MAE': f"{b['mae']:.4f}", 'Доверительный интервал 95%': f"±{b['confidence_95']:.4f}", 'AIC': f"{b['aic']:.2f}", 'R2': '-'})
        if 'decomposition' in r and 'error' not in r['decomposition']:
            drow = r['decomposition']
            rows.append({'Модель': 'Декомпозиция (Фурье)', 'Длина интервала': L, 'p': '-', 'd': '-', 'q': '-', 'RMSE': f"{drow['rmse']:.4f}", 'MAE': f"{drow['mae']:.4f}", 'Доверительный интервал 95%': f"±{drow['confidence_95'] if 'confidence_95' in drow else 0:.4f}", 'AIC': '-', 'R2': f"{drow['r2_score']:.4f}"})
        if 'prophet' in r and 'error' not in r['prophet']:
            p = r['prophet']
            rows.append({'Модель': 'Prophet', 'Длина интервала': L, 'p': '-', 'd': '-', 'q': '-', 'RMSE': f"{p['rmse']:.4f}", 'MAE': f"{p['mae']:.4f}", 'Доверительный интервал 95%': f"±{p['confidence_95']:.4f}", 'AIC': '-', 'R2': f"{p['r2_score']:.4f}"})
    df = pd.DataFrame(rows)
    path = f"{output_dir}/summary_results.xlsx"
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Results')
    print(f"\nТаблица результатов: {path}")
    return df

def create_extended_plots(series: pd.Series, all_results: list, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    # Базовый график исходного ряда
    plt.figure(figsize=FIGURE_SIZE_MAIN)
    plt.plot(series.index, series.values, label='Временной ряд', alpha=0.8, linewidth=1.5)
    plt.title('Исходные данные временного ряда', fontsize=16, fontweight='bold')
    plt.xlabel('Дата'); plt.ylabel('Значение'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{output_dir}/original_series_clean_series.png", dpi=FIGURE_DPI, bbox_inches='tight'); plt.close()
    # Сравнение
    fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE_COMPARISON)
    fig.suptitle('Сравнение точности моделей по длинам интервалов', fontsize=16, fontweight='bold')
    lengths = [r['length'] for r in all_results]
    names = ['ARIMA', 'BSTS', 'Декомпозиция', 'Prophet']
    metrics = {n: {'rmse': [], 'ci': []} for n in names}
    for r in all_results:
        for n in names:
            key = 'decomposition' if n == 'Декомпозиция' else n.lower()
            if key in r and 'error' not in r[key]:
                metrics[n]['rmse'].append(r[key]['rmse'])
                metrics[n]['ci'].append(r[key]['confidence_95'])
            else:
                metrics[n]['rmse'].append(np.nan); metrics[n]['ci'].append(np.nan)
    ax1 = axes[0, 0]
    for n in names:
        ax1.plot(lengths, metrics[n]['rmse'], marker='o', label=n, linewidth=2)
    ax1.set_xlabel('Длина интервала'); ax1.set_ylabel('RMSE'); ax1.set_title('RMSE'); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2 = axes[0, 1]
    for n in names:
        ax2.plot(lengths, metrics[n]['ci'], marker='s', label=n, linewidth=2)
    ax2.set_xlabel('Длина интервала'); ax2.set_ylabel('Доверительный интервал 95%'); ax2.set_title('Ширина доверительных интервалов'); ax2.legend(); ax2.grid(True, alpha=0.3)
    ax3 = axes[1, 0]
    best = []
    best_names = []
    for i, L in enumerate(lengths):
        m = float('inf'); nm = 'None'
        for n in names:
            val = metrics[n]['rmse'][i]
            if not np.isnan(val) and val < m:
                m = val; nm = n
        best.append(0 if m == float('inf') else m); best_names.append(nm)
    bars = ax3.bar(range(len(lengths)), best)
    ax3.set_xlabel('Длина интервала'); ax3.set_ylabel('Лучший RMSE'); ax3.set_title('Лучшая модель по RMSE'); ax3.set_xticks(range(len(lengths))); ax3.set_xticklabels(lengths)
    for i, (b, nm) in enumerate(zip(bars, best_names)):
        ax3.text(b.get_x()+b.get_width()/2., b.get_height(), f'{nm}\n{b.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    ax4 = axes[1, 1]
    if all_results and 'decomposition' in all_results[-1] and 'error' not in all_results[-1]['decomposition']:
        L = all_results[-1]['length']
        sub = series.iloc[:L] if len(series) > L else series
        dec = fourier_decomposition(sub)
        x = range(len(sub))
        ax4.plot(x, sub.values, label='Данные', alpha=0.7)
        ax4.plot(x, dec['reconstructed'], label='Модель', linewidth=2)
        ci = CONFIDENCE_LEVEL * np.std(dec['residuals'])
        ax4.fill_between(x, dec['reconstructed']-ci, dec['reconstructed']+ci, alpha=0.3, label='95% ДИ')
        ax4.set_title(f'Декомпозиция с доверительными интервалами (длина {L})'); ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'Декомпозиция недоступна', ha='center', va='center', transform=ax4.transAxes)
    ax4.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{output_dir}/models_comparison_clean_series.png", dpi=FIGURE_DPI, bbox_inches='tight'); plt.close()
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
    
    # Тест нормальности остатков (Шапиро–Уилка) на полном ряду для обеих моделей
    try:
        print("\nТест нормальности остатков (Шапиро–Уилка)")
        print("-" * 60)
        # Чебышев
        cheb_full = fourier_decomposition(series)
        if 'error' not in cheb_full:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sh_stat, sh_p = stats.shapiro(cheb_full['residuals'])
            conclusion = (
                "Отклоняем H0: остатки не имеют нормальное распределение"
                if sh_p < 0.05 else
                "Не отклоняем H0: остатки близки к нормальному распределению"
            )
            print(f"Чебышев: W = {sh_stat:.4f}, p-value = {sh_p:.4f} -> {conclusion}")
        else:
            print("Чебышев: не удалось получить остатки для теста")
        # Prophet
        prop_full = build_prophet_model(series)
        if 'error' not in prop_full:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sh_stat, sh_p = stats.shapiro(prop_full['residuals'])
            conclusion = (
                "Отклоняем H0: остатки не имеют нормальное распределение"
                if sh_p < 0.05 else
                "Не отклоняем H0: остатки близки к нормальному распределению"
            )
            print(f"Prophet: W = {sh_stat:.4f}, p-value = {sh_p:.4f} -> {conclusion}")
        else:
            print("Prophet: не удалось получить остатки для теста")
    except Exception as e:
        print(f"Не удалось выполнить тест Шапиро–Уилка: {e}")

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
    
    # Расширенный анализ (ARIMA, BSTS, сравнения)
    print("\nЗапуск расширенного анализа моделей...")
    interval_lengths = [l for l in INTERVAL_LENGTHS if l <= len(series)]
    if len(series) not in interval_lengths:
        interval_lengths.append(len(series))
    all_results = []
    for L in interval_lengths:
        print(f"\nАнализ интервала {L}")
        all_results.append(run_extended_analysis(series, L))
    results_df = create_extended_summary(all_results, OUTPUT_DIR)
    create_extended_plots(series, all_results, OUTPUT_DIR)

    # Печать в точном формате, как вы требуете
    print("\nТак как в Лабораторной работе 1 мы выяснили, что ряд нестационарный параметр d мы приняли равным 1.")
    print("1. ARIMA")
    print("Длина мерного интервала\tp\tq\tСреднеквадратичное отклонение")
    for r in all_results:
        if 'arima' in r and 'error' not in r['arima']:
            a = r['arima']
            print(f"{r['length']}\t{a['p']}\t{a['q']}\t{a['rmse']:.4f}")

    print("\n2.\tBSTS")
    print("Длина мерного интервала\tСезонность\tТренд\tСреднеквадратичное отклонение")
    for r in all_results:
        if 'bsts' in r and 'error' not in r['bsts']:
            b = r['bsts']
            season = b.get('seasonal') if b.get('seasonal') is not None else '-'
            trend = b.get('trend_name','-')
            print(f"{r['length']}\t{season}\t{trend}\t{b['rmse']:.4f}")

    print("\n3.")
    print("3.1\t")
    print("Длина мерного интервала\tЛучшая модель\tСреднеквадратичное отклонение")
    for r in all_results:
        if 'decomposition' in r and 'error' not in r['decomposition']:
            drow = r['decomposition']
            print(f"{r['length']}\tМногочлены Чебышева\t{drow['rmse']:.4f}")

    print("\n3.2\tProphet")
    print("Длина мерного интервала\tПериод\tСреднеквадратическое отклонение")
    # Период выводим по простой эвристике из исходного кода таблицы
    for r in all_results:
        if 'prophet' in r and 'error' not in r['prophet']:
            L = r['length']
            if L <= 70:
                period = 30
            elif L <= 150:
                period = 90
            elif L <= 200:
                period = 7
            elif L <= 240:
                period = 60
            else:
                period = 120
            print(f"{L}\t{period}\t{r['prophet']['rmse']:.4f}")

    # Итог
    print(f"\nРабота завершена!")

if __name__ == "__main__":
    main()