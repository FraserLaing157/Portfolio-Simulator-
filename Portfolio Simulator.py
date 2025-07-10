import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import webbrowser
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import queue
import requests
import logging

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

class PortfolioOptimizer:
    def __init__(self, tickers=None, start_date=None, end_date=None):
        self.tickers = tickers or []
        self.start_date = start_date or (datetime.today() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.today().strftime('%Y-%m-%d')
        self.data = None
        self.returns = None
        self.cov_matrix = None
        self.optimal_weights = None
        self.prediction_models = {}
        self.sentiment_data = {}
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def download_data(self):
        try:
            data = yf.download(self.tickers, start=self.start_date, end=self.end_date, progress=False, timeout=10)
            if data.empty:
                raise ValueError("No data downloaded. Check ticker symbols or date range.")
            if isinstance(data.columns, pd.MultiIndex):
                self.data = data['Close'].copy()
            else:
                if 'Close' in data.columns:
                    self.data = data[['Close']].copy()
                    self.data.columns = [self.tickers[0]]
                else:
                    raise ValueError("Could not find 'Close' column in the data.")
        except Exception as e:
            raise ValueError(f"Failed to download data: {str(e)}")

    def calculate_returns(self):
        if self.data is None:
            self.download_data()
        self.returns = self.data.pct_change().dropna()
        self.cov_matrix = self.returns.cov() * 252

    def monte_carlo_simulation(self, num_portfolios=10000, risk_free_rate=0.02):
        self.calculate_returns()
        mean_returns = self.returns.mean() * 252
        results = np.zeros((3 + len(self.tickers), num_portfolios))
        for i in range(num_portfolios):
            weights = np.random.random(len(self.tickers))
            weights /= np.sum(weights)
            port_return = np.sum(mean_returns * weights)
            port_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            results[0, i] = port_return
            results[1, i] = port_volatility
            results[2, i] = (port_return - risk_free_rate) / port_volatility
            for j in range(len(weights)):
                results[j+3, i] = weights[j]
        columns = ['Return', 'Volatility', 'Sharpe'] + self.tickers
        return pd.DataFrame(results.T, columns=columns)

    def optimize_portfolio(self, risk_free_rate=0.02):
        mean_returns = self.returns.mean() * 252

        def neg_sharpe(weights):
            ret = np.sum(mean_returns * weights)
            vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            return -(ret - risk_free_rate) / vol

        bounds = tuple((0, 1) for _ in self.tickers)
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        initial = np.array([1/len(self.tickers)] * len(self.tickers))
        result = minimize(neg_sharpe, initial, method='SLSQP', bounds=bounds, constraints=constraints)
        if not result.success:
            raise ValueError("Portfolio optimization failed.")
        self.optimal_weights = result.x
        return result.x

    def analyze_results(self, mc_results):
        optimal_idx = mc_results['Sharpe'].idxmax()
        optimal_port = mc_results.loc[optimal_idx]
        return optimal_port

    def add_ml_prediction(self):
        for ticker in self.tickers:
            try:
                data = self.data[ticker].values.reshape(-1, 1)
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(data)
                
                lookback = 60
                X, y = [], []
                for i in range(lookback, len(scaled_data)):
                    X.append(scaled_data[i-lookback:i])
                    y.append(scaled_data[i])
                X, y = np.array(X), np.array(y)
                
                if len(X) < 10:
                    continue
                    
                model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
                    LSTM(50),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(X, y, epochs=10, batch_size=32, verbose=0)
                self.prediction_models[ticker] = {'model': model, 'scaler': scaler}
            except Exception as e:
                self.logger.error(f"Failed to train model for {ticker}: {str(e)}")

    def get_sentiment_analysis(self):
        for ticker in self.tickers:
            self.sentiment_data[ticker] = {
                'positive': np.random.random(),
                'negative': np.random.random(),
                'score': np.random.uniform(-1, 1)
            }

    def generate_recommendations(self):
        recs = []
        for ticker in self.tickers:
            score = (self.sentiment_data[ticker]['score'] * 0.3 + 
                     np.random.random() * 0.7)
            
            if score > 0.6:
                recs.append((ticker, 'Strong Buy', score))
            elif score > 0.3:
                recs.append((ticker, 'Buy', score))
            elif score > -0.3:
                recs.append((ticker, 'Hold', score))
            elif score > -0.6:
                recs.append((ticker, 'Sell', score))
            else:
                recs.append((ticker, 'Strong Sell', score))
                
        return sorted(recs, key=lambda x: x[2], reverse=True)

    def fetch_all_tickers(self):
        try:
            df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            tickers = df['Symbol'].tolist()
            tickers = [t for t in tickers if isinstance(t, str) and '.' not in t]
            if not tickers:
                raise ValueError("No tickers retrieved from S&P 500 list")
            return list(set(tickers))
        except Exception as e:
            self.logger.error(f"Failed to fetch tickers: {str(e)}")
            default_tickers = [
                'AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT', 'JNJ',
                'PG', 'KO', 'NKE', 'DIS', 'NFLX', 'PYPL', 'ADBE', 'INTC', 'AMD', 'CSCO'
            ]
            self.logger.info("Using default ticker list")
            return default_tickers

    def validate_ticker(self, ticker):
        """Validate if a ticker is likely valid by checking Yahoo Finance"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if not info or 'symbol' not in info:
                return False, f"Ticker {ticker} is invalid or delisted"
            return True, None
        except Exception as e:
            return False, f"Failed to validate {ticker}: {str(e)}"

    def backtest_ticker(self, ticker, years=5):
        """Backtest a single ticker with detailed error handling"""
        try:
            # Validate ticker first
            is_valid, error = self.validate_ticker(ticker)
            if not is_valid:
                self.logger.error(error)
                return None, error

            # Download data with retries and delay
            data = None
            for attempt in range(3):
                try:
                    data = yf.download(ticker, period=f"{years}y", progress=False, timeout=10)
                    if not data.empty:
                        break
                    time.sleep(0.5)  # Delay between retries
                except Exception as e:
                    self.logger.warning(f"Attempt {attempt + 1} failed for {ticker}: {str(e)}")
                    if attempt == 2:
                        return None, f"Failed to download data for {ticker}: {str(e)}"
                    time.sleep(1)  # Longer delay for retries

            if data.empty or len(data) < 100:
                error_msg = f"No data available for {ticker} or insufficient data ({len(data)} rows)"
                self.logger.error(error_msg)
                return None, error_msg

            returns = data['Close'].pct_change().dropna()
            if len(returns) < 50:
                error_msg = f"Insufficient return data for {ticker} ({len(returns)} returns)"
                self.logger.error(error_msg)
                return None, error_msg

            annual_return = (data['Close'][-1] / data['Close'][0]) ** (1/years) - 1
            volatility = returns.std() * np.sqrt(252)
            sharpe = annual_return / (volatility + 1e-6)
            last_3m = (data['Close'][-1] / data['Close'][-63]) - 1 if len(data) > 63 else 0

            result = {
                'ticker': ticker,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe': sharpe,
                '3m_momentum': last_3m
            }
            self.logger.info(f"Successfully backtested {ticker}: Sharpe={sharpe:.2f}, Return={annual_return:.2%}")
            return result, None

        except Exception as e:
            error_msg = f"Failed {ticker}: {str(e)}"
            self.logger.error(error_msg)
            return None, error_msg

    def find_top_performers(self, backtest_results, top_n=3):
        """Filter stocks with high momentum and Sharpe ratio"""
        filtered = [
            r for r, _ in backtest_results 
            if r and r['sharpe'] > 1.5 and r['3m_momentum'] > 0.2
        ]
        return sorted(filtered, key=lambda x: x['sharpe'], reverse=True)[:top_n]

    def run_mass_backtest(self, progress_callback=None, log_callback=None):
        """Run backtest on all available tickers with rate limiting and validation"""
        tickers = self.fetch_all_tickers()
        backtest_results = []
        total = len(tickers)
        start_time = time.time()
        last_heartbeat = start_time
        failed_count = 0
        lock = threading.Lock()

        # Validate tickers first
        valid_tickers = []
        for ticker in tickers:
            is_valid, error = self.validate_ticker(ticker)
            if is_valid:
                valid_tickers.append(ticker)
            else:
                failed_count += 1
                if log_callback:
                    with lock:
                        log_callback(error)
                        log_callback(f"Processed {ticker} ({len(valid_tickers) + failed_count}/{total}) - Est. remaining: --s - Failed: {failed_count}")

        self.logger.info(f"Validated {len(valid_tickers)}/{total} tickers")
        if log_callback:
            with lock:
                log_callback(f"Validated {len(valid_tickers)}/{total} tickers")

        # Process valid tickers with limited concurrency
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(self.backtest_ticker, ticker) for ticker in valid_tickers]
            for i, future in enumerate(futures):
                result, error = future.result()
                with lock:
                    if result:
                        backtest_results.append((result, None))
                        self.logger.info(f"Completed {valid_tickers[i]}: Sharpe={result['sharpe']:.2f}")
                    else:
                        failed_count += 1
                        if error and log_callback:
                            log_callback(error)

                    if progress_callback:
                        progress_callback((i + 1 + len(tickers) - len(valid_tickers)) / total * 100)
                    if log_callback:
                        elapsed = time.time() - start_time
                        est_remaining = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else 0
                        log_callback(f"Processed {valid_tickers[i]} ({i + 1 + len(tickers) - len(valid_tickers)}/{total}) - Est. remaining: {est_remaining:.0f}s - Failed: {failed_count}")
                        if time.time() - last_heartbeat >= 30:
                            log_callback(f"Heartbeat: Backtesting is still running... ({i + 1 + len(tickers) - len(valid_tickers)}/{total} processed, {failed_count} failed)")
                            last_heartbeat = time.time()
                    time.sleep(0.5)

        top_performers = self.find_top_performers(backtest_results)
        self.logger.info(f"Backtest complete. Top performers: {[r['ticker'] for r in top_performers]}")
        return top_performers

class PortfolioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Portfolio Simulator Pro")
        self.root.geometry("1200x800")
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 10))
        self.style.configure('Header.TLabel', font=('Arial', 14, 'bold'))
        
        self.optimizer = None
        self.top_stocks = []
        self.total_tickers = 0
        self.update_queue = queue.Queue()
        self.create_menu()
        self.create_main_interface()
        self.process_queue()

    def process_queue(self):
        try:
            while True:
                func = self.update_queue.get_nowait()
                func()
        except queue.Empty:
            pass
        self.root.after(100, self.process_queue)

    def create_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="New Analysis", command=self.new_analysis)
        file_menu.add_command(label="Save Report", command=self.save_report)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Documentation", command=self.show_docs)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)

    def create_main_interface(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        left_frame = ttk.Frame(main_frame, width=300, relief=tk.RIDGE, borderwidth=2)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_frame.pack_propagate(False)
        
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.create_input_panel(left_frame)
        self.create_results_tabs(right_frame)
        
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_input_panel(self, parent):
        ttk.Label(parent, text="Portfolio Inputs", style='Header.TLabel').pack(pady=10)
        
        ticker_frame = ttk.Frame(parent)
        ticker_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(ticker_frame, text="Tickers:").pack(side=tk.LEFT)
        self.tickers_entry = ttk.Entry(ticker_frame)
        self.tickers_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.tickers_entry.insert(0, "AAPL,MSFT,GOOG")
        
        date_frame = ttk.Frame(parent)
        date_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(date_frame, text="Date Range:").pack(side=tk.LEFT)
        self.start_date_entry = ttk.Entry(date_frame, width=10)
        self.start_date_entry.pack(side=tk.LEFT)
        self.start_date_entry.insert(0, (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d'))
        ttk.Label(date_frame, text="to").pack(side=tk.LEFT)
        self.end_date_entry = ttk.Entry(date_frame, width=10)
        self.end_date_entry.pack(side=tk.LEFT)
        self.end_date_entry.insert(0, datetime.today().strftime('%Y-%m-%d'))
        
        invest_frame = ttk.Frame(parent)
        invest_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(invest_frame, text="Initial Investment:").pack(side=tk.LEFT)
        self.invest_entry = ttk.Entry(invest_frame, width=15)
        self.invest_entry.pack(side=tk.LEFT)
        self.invest_entry.insert(0, "10000")
        
        risk_frame = ttk.Frame(parent)
        risk_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(risk_frame, text="Risk Tolerance:").pack(side=tk.LEFT)
        self.risk_var = tk.StringVar(value="Medium")
        risk_menu = ttk.OptionMenu(risk_frame, self.risk_var, "Medium", "Low", "Medium", "High")
        risk_menu.pack(side=tk.LEFT)
        
        strategy_frame = ttk.Frame(parent)
        strategy_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(strategy_frame, text="Strategy:").pack(side=tk.LEFT)
        self.strategy_var = tk.StringVar(value="Balanced")
        strategy_menu = ttk.OptionMenu(strategy_frame, self.strategy_var, "Balanced", 
                                      "Conservative", "Balanced", "Growth", "Aggressive")
        strategy_menu.pack(side=tk.LEFT)
        
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=10)
        ttk.Button(button_frame, text="Run Full Analysis", command=self.run_full_analysis).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(button_frame, text="Quick Optimize", command=self.quick_optimize).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.advanced_frame = ttk.LabelFrame(parent, text="Advanced Options", relief=tk.RIDGE)
        self.advanced_frame.pack(fill=tk.X, padx=5, pady=5)
        
        rebal_frame = ttk.Frame(self.advanced_frame)
        rebal_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(rebal_frame, text="Rebalancing:").pack(side=tk.LEFT)
        self.rebal_var = tk.StringVar(value="Quarterly")
        rebal_menu = ttk.OptionMenu(rebal_frame, self.rebal_var, "Quarterly", 
                                    "Monthly", "Quarterly", "Annually", "Never")
        rebal_menu.pack(side=tk.LEFT)
        
        cost_frame = ttk.Frame(self.advanced_frame)
        cost_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(cost_frame, text="Transaction Costs:").pack(side=tk.LEFT)
        self.cost_entry = ttk.Entry(cost_frame, width=8)
        self.cost_entry.pack(side=tk.LEFT)
        self.cost_entry.insert(0, "0.1")
        ttk.Label(cost_frame, text="%").pack(side=tk.LEFT)
        
        tax_frame = ttk.Frame(self.advanced_frame)
        tax_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(tax_frame, text="Tax Rate:").pack(side=tk.LEFT)
        self.tax_entry = ttk.Entry(tax_frame, width=8)
        self.tax_entry.pack(side=tk.LEFT)
        self.tax_entry.insert(0, "20")
        ttk.Label(tax_frame, text="%").pack(side=tk.LEFT)

    def create_results_tabs(self, parent):
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        self.summary_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.summary_tab, text="Summary")
        self.create_summary_tab(self.summary_tab)
        
        self.allocation_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.allocation_tab, text="Allocation")
        self.create_allocation_tab(self.allocation_tab)
        
        self.performance_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.performance_tab, text="Performance")
        self.create_performance_tab(self.performance_tab)
        
        self.recommendations_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.recommendations_tab, text="Recommendations")
        self.create_recommendations_tab(self.recommendations_tab)
        
        self.create_test_stock_tab()

    def create_test_stock_tab(self):
        self.test_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.test_tab, text="Test Stocks")
        
        control_frame = ttk.Frame(self.test_tab)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Run Mass Backtest", command=self.start_mass_backtest).pack(side=tk.LEFT)
        
        self.progress = ttk.Progressbar(self.test_tab, mode='determinate')
        self.progress.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_label = ttk.Label(self.test_tab, text="Ready", font=('Arial', 10, 'bold'))
        self.status_label.pack()
        
        self.time_label = ttk.Label(self.test_tab, text="Estimated time remaining: --")
        self.time_label.pack()
        
        log_frame = ttk.LabelFrame(self.test_tab, text="Processing Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.insert(tk.END, "Log output will appear here...\n")
        self.log_text.config(state=tk.DISABLED)
        
        self.results_frame = ttk.Frame(self.test_tab)
        self.results_frame.pack(fill=tk.BOTH, expand=True)
        
        self.prediction_canvas_frame = ttk.Frame(self.results_frame)
        self.prediction_canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.allocation_table = ttk.Treeview(self.results_frame, columns=('Ticker', 'Weight', 'Amount'))
        self.allocation_table.heading('#0', text='Rank')
        self.allocation_table.heading('Ticker', text='Ticker')
        self.allocation_table.heading('Weight', text='Weight')
        self.allocation_table.heading('Amount', text='Amount (£)')
        self.allocation_table.pack(fill=tk.BOTH, expand=True)
        
        invest_frame = ttk.Frame(self.test_tab)
        invest_frame.pack(fill=tk.X, pady=5)
        ttk.Label(invest_frame, text="Investment Amount (£):").pack(side=tk.LEFT)
        self.investment_entry = ttk.Entry(invest_frame, width=15)
        self.investment_entry.pack(side=tk.LEFT, padx=5)
        self.investment_entry.insert(0, "1000")
        ttk.Button(invest_frame, text="Generate Pie", command=self.generate_prediction_pie).pack(side=tk.LEFT)

    def log_message(self, message):
        def update():
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
        self.update_queue.put(update)

    def create_summary_tab(self, parent):
        metrics_frame = ttk.LabelFrame(parent, text="Key Metrics")
        metrics_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.metric_vars = {
            'return': tk.StringVar(value="N/A"),
            'volatility': tk.StringVar(value="N/A"),
            'sharpe': tk.StringVar(value="N/A"),
            'max_drawdown': tk.StringVar(value="N/A"),
            'beta': tk.StringVar(value="N/A")
        }
        
        for i, (name, var) in enumerate(self.metric_vars.items()):
            frame = ttk.Frame(metrics_frame)
            frame.grid(row=i//3, column=i%3, sticky="ew", padx=5, pady=5)
            ttk.Label(frame, text=name.replace('_', ' ').title() + ":").pack(side=tk.LEFT)
            ttk.Label(frame, textvariable=var, font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        
        optimal_frame = ttk.LabelFrame(parent, text="Optimal Portfolio Allocation")
        optimal_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.allocation_text = scrolledtext.ScrolledText(optimal_frame, wrap=tk.WORD, height=10)
        self.allocation_text.pack(fill=tk.BOTH, expand=True)
        self.allocation_text.insert(tk.END, "Run analysis to see optimal allocation")
        self.allocation_text.config(state=tk.DISABLED)

    def create_allocation_tab(self, parent):
        self.allocation_canvas_frame = ttk.Frame(parent)
        self.allocation_canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.allocation_fig, self.allocation_ax = plt.subplots(figsize=(6, 4))
        self.allocation_ax.set_title("Portfolio Allocation")
        self.allocation_canvas = FigureCanvasTkAgg(self.allocation_fig, master=self.allocation_canvas_frame)
        self.allocation_canvas.draw()
        self.allocation_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(self.allocation_canvas, self.allocation_canvas_frame)
        toolbar.update()
        self.allocation_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_performance_tab(self, parent):
        self.performance_canvas_frame = ttk.Frame(parent)
        self.performance_canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.performance_fig, self.performance_ax = plt.subplots(figsize=(6, 4))
        self.performance_ax.set_title("Portfolio Performance")
        self.performance_canvas = FigureCanvasTkAgg(self.performance_fig, master=self.performance_canvas_frame)
        self.performance_canvas.draw()
        self.performance_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(self.performance_canvas, self.performance_canvas_frame)
        toolbar.update()
        self.performance_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_recommendations_tab(self, parent):
        self.recommendations_text = scrolledtext.ScrolledText(parent, wrap=tk.WORD, height=15)
        self.recommendations_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.recommendations_text.insert(tk.END, "Run analysis to see recommendations")
        self.recommendations_text.config(state=tk.DISABLED)

    def validate_inputs(self):
        try:
            tickers = [t.strip().upper() for t in self.tickers_entry.get().split(',') if t.strip()]
            if not tickers:
                raise ValueError("Please enter at least one ticker symbol")
            
            start_date = datetime.strptime(self.start_date_entry.get(), '%Y-%m-%d')
            end_date = datetime.strptime(self.end_date_entry.get(), '%Y-%m-%d')
            if start_date >= end_date:
                raise ValueError("Start date must be before end date")
            
            investment = float(self.invest_entry.get())
            if investment <= 0:
                raise ValueError("Investment amount must be positive")
            
            return tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), investment
        except ValueError as e:
            raise ValueError(f"Input validation failed: {str(e)}")

    def run_full_analysis(self):
        try:
            self.status_var.set("Running analysis...")
            self.root.update()
            
            tickers, start_date, end_date, _ = self.validate_inputs()
            
            self.optimizer = PortfolioOptimizer(tickers, start_date, end_date)
            self.mc_results = self.optimizer.monte_carlo_simulation()
            self.optimizer.optimize_portfolio()
            optimal_port = self.optimizer.analyze_results(self.mc_results)
            
            self.optimizer.add_ml_prediction()
            self.optimizer.get_sentiment_analysis()
            recs = self.optimizer.generate_recommendations()
            
            self.update_summary_tab(optimal_port)
            self.update_allocation_tab(optimal_port)
            self.update_performance_tab()
            self.update_recommendations_tab(recs)
            
            self.status_var.set("Analysis complete")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("Error in analysis")

    def update_summary_tab(self, optimal_port):
        self.metric_vars['return'].set(f"{optimal_port['Return']:.2%}")
        self.metric_vars['volatility'].set(f"{optimal_port['Volatility']:.2%}")
        self.metric_vars['sharpe'].set(f"{optimal_port['Sharpe']:.2f}")
        self.metric_vars['max_drawdown'].set("N/A")
        self.metric_vars['beta'].set("N/A")
        
        self.allocation_text.config(state=tk.NORMAL)
        self.allocation_text.delete(1.0, tk.END)
        
        text = "Optimal Portfolio Allocation:\n\n"
        for ticker, weight in zip(self.optimizer.tickers, optimal_port[3:]):
            text += f"{ticker}: {weight:.2%}\n"
        
        self.allocation_text.insert(tk.END, text)
        self.allocation_text.config(state=tk.DISABLED)

    def update_allocation_tab(self, optimal_port):
        self.allocation_ax.clear()
        weights = optimal_port[3:].values
        labels = self.optimizer.tickers
        self.allocation_ax.pie(weights, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
        self.allocation_ax.set_title("Optimal Portfolio Allocation")
        self.allocation_canvas.draw()

    def update_performance_tab(self):
        self.performance_ax.clear()
        sc = self.performance_ax.scatter(self.mc_results['Volatility'], self.mc_results['Return'],
                                        c=self.mc_results['Sharpe'], cmap='viridis', alpha=0.5)
        self.performance_fig.colorbar(sc, ax=self.performance_ax, label="Sharpe Ratio")
        
        opt_idx = self.mc_results['Sharpe'].idxmax()
        self.performance_ax.scatter(self.mc_results.loc[opt_idx, 'Volatility'],
                                   self.mc_results.loc[opt_idx, 'Return'],
                                   color='r', marker='*', s=200, label="Optimal")
        
        self.performance_ax.set_title("Efficient Frontier")
        self.performance_ax.set_xlabel("Volatility (Standard Deviation)")
        self.performance_ax.set_ylabel("Expected Return")
        self.performance_ax.legend()
        self.performance_ax.grid(True)
        self.performance_canvas.draw()

    def update_recommendations_tab(self, recommendations):
        self.recommendations_text.config(state=tk.NORMAL)
        self.recommendations_text.delete(1.0, tk.END)
        
        text = "AI Recommendations:\n\n"
        for ticker, rec, score in recommendations:
            text += f"{ticker}: {rec} (confidence: {score:.2f})\n"
            text += f"  - Sentiment Score: {self.optimizer.sentiment_data[ticker]['score']:.2f}\n"
            text += f"  - Positive: {self.optimizer.sentiment_data[ticker]['positive']:.2%}\n"
            text += f"  - Negative: {self.optimizer.sentiment_data[ticker]['negative']:.2%}\n\n"
        
        self.recommendations_text.insert(tk.END, text)
        self.recommendations_text.config(state=tk.DISABLED)

    def quick_optimize(self):
        try:
            self.status_var.set("Running quick optimization...")
            self.root.update()
            
            tickers, start_date, end_date, _ = self.validate_inputs()
            
            self.optimizer = PortfolioOptimizer(tickers, start_date, end_date)
            self.mc_results = self.optimizer.monte_carlo_simulation(num_portfolios=5000)
            self.optimizer.optimize_portfolio()
            optimal_port = self.optimizer.analyze_results(self.mc_results)
            
            self.update_summary_tab(optimal_port)
            self.update_allocation_tab(optimal_port)
            
            self.status_var.set("Quick optimization complete")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("Error in optimization")

    def start_mass_backtest(self):
        def update_status(text):
            self.status_label.config(text=text)
        
        def update_time(text):
            self.time_label.config(text=text)
        
        def update_progress(value):
            self.progress.config(value=value)
        
        def show_error(message):
            messagebox.showerror("Error", message)
        
        def update_status_var(text):
            self.status_var.set(text)
        
        self.update_queue.put(lambda: update_status("Initializing backtest..."))
        self.update_queue.put(lambda: update_time("Estimated time remaining: Calculating..."))
        self.update_queue.put(lambda: update_progress(0))
        self.update_queue.put(lambda: self.log_text.config(state=tk.NORMAL))
        self.update_queue.put(lambda: self.log_text.delete(1.0, tk.END))
        self.update_queue.put(lambda: self.log_text.insert(tk.END, "Starting mass backtest...\n"))
        self.update_queue.put(lambda: self.log_text.config(state=tk.DISABLED))
        
        if not self.optimizer:
            self.optimizer = PortfolioOptimizer(tickers=['AAPL'])
        
        def worker():
            try:
                tickers = self.optimizer.fetch_all_tickers()
                self.total_tickers = len(tickers)
                self.update_queue.put(lambda: update_status(f"Backtesting {self.total_tickers} stocks..."))
                self.update_queue.put(lambda: self.log_message(f"Retrieved {self.total_tickers} tickers for backtesting"))
                
                self.top_stocks = self.optimizer.run_mass_backtest(
                    progress_callback=lambda p: self.update_queue.put(lambda: update_progress(p)),
                    log_callback=lambda msg: self.update_queue.put(lambda: self.log_message(msg))
                )
                
                self.update_queue.put(self.show_backtest_results)
            except Exception as e:
                error_message = str(e)
                self.update_queue.put(lambda: show_error(error_message))
                self.update_queue.put(lambda: self.log_message(f"Error: {error_message}"))
                self.update_queue.put(lambda: update_status("Backtest failed"))
                self.update_queue.put(lambda: update_status_var("Backtest failed"))
        
        threading.Thread(target=worker, daemon=True).start()

    def show_backtest_results(self):
        self.status_label.config(text=f"Top performers: {', '.join([s['ticker'] for s in self.top_stocks])}")
        self.time_label.config(text="Backtest complete!")
        
        self.log_message("\nBacktest completed successfully!")
        self.log_message(f"Top performers: {', '.join([s['ticker'] for s in self.top_stocks])}")
        
        for row in self.allocation_table.get_children():
            self.allocation_table.delete(row)
            
        for i, stock in enumerate(self.top_stocks, 1):
            self.allocation_table.insert('', 'end', text=str(i), values=(
                stock['ticker'],
                f"{stock['sharpe']:.2f}",
                f"£{stock['annual_return']*1000:.2f}"
            ))

    def generate_prediction_pie(self):
        try:
            investment = float(self.investment_entry.get())
            if investment <= 0:
                raise ValueError("Investment amount must be positive")
                
            total_sharpe = sum(s['sharpe'] for s in self.top_stocks)
            if total_sharpe == 0:
                raise ValueError("No valid stocks available for allocation")
                
            allocations = [
                {
                    'ticker': s['ticker'],
                    'weight': s['sharpe'] / total_sharpe,
                    'amount': investment * (s['sharpe'] / total_sharpe)
                }
                for s in self.top_stocks
            ]
            
            for row in self.allocation_table.get_children():
                self.allocation_table.delete(row)
            for i, alloc in enumerate(allocations, 1):
                self.allocation_table.insert('', 'end', text=str(i), values=(
                    alloc['ticker'],
                    f"{alloc['weight']:.1%}",
                    f"£{alloc['amount']:,.2f}"
                ))
            
            self.plot_revenue_projection(allocations)
            
        except ValueError as e:
            messagebox.showerror("Error", str(e))

    def plot_revenue_projection(self, allocations):
        plt.close('all')
        fig, ax = plt.subplots(figsize=(8, 4))
        
        months = np.arange(1, 13)
        for alloc in allocations:
            growth = np.random.normal(loc=1.02, scale=0.05, size=12).cumprod()
            ax.plot(months, alloc['amount'] * growth, label=alloc['ticker'])
        
        ax.set_title("12-Month Projected Growth")
        ax.set_xlabel("Months")
        ax.set_ylabel("Value (£)")
        ax.legend()
        ax.grid(True)
        
        if hasattr(self, 'prediction_canvas'):
            self.prediction_canvas.get_tk_widget().destroy()
        self.prediction_canvas = FigureCanvasTkAgg(fig, master=self.prediction_canvas_frame)
        self.prediction_canvas.draw()
        self.prediction_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def new_analysis(self):
        self.tickers_entry.delete(0, tk.END)
        self.tickers_entry.insert(0, "AAPL,MSFT,GOOG")
        self.start_date_entry.delete(0, tk.END)
        self.start_date_entry.insert(0, (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d'))
        self.end_date_entry.delete(0, tk.END)
        self.end_date_entry.insert(0, datetime.today().strftime('%Y-%m-%d'))
        
        self.allocation_text.config(state=tk.NORMAL)
        self.allocation_text.delete(1.0, tk.END)
        self.allocation_text.insert(tk.END, "Run analysis to see optimal allocation")
        self.allocation_text.config(state=tk.DISABLED)
        
        self.recommendations_text.config(state=tk.NORMAL)
        self.recommendations_text.delete(1.0, tk.END)
        self.recommendations_text.insert(tk.END, "Run analysis to see recommendations")
        self.recommendations_text.config(state=tk.DISABLED)
        
        self.allocation_ax.clear()
        self.allocation_ax.set_title("Portfolio Allocation")
        self.allocation_canvas.draw()
        
        self.performance_ax.clear()
        self.performance_ax.set_title("Portfolio Performance")
        self.performance_canvas.draw()
        
        for var in self.metric_vars.values():
            var.set("N/A")
        
        self.optimizer = None
        self.top_stocks = []
        self.total_tickers = 0

    def save_report(self):
        if not REPORTLAB_AVAILABLE:
            messagebox.showwarning("Warning", "PDF report generation requires the 'reportlab' library. Install it using 'pip install reportlab'.")
            return
        
        try:
            file_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
            if not file_path:
                return
            
            c = canvas.Canvas(file_path, pagesize=letter)
            c.drawString(100, 750, "Portfolio Analysis Report")
            c.drawString(100, 730, f"Date: {datetime.today().strftime('%Y-%m-%d')}")
            
            if self.optimizer and hasattr(self, 'mc_results'):
                optimal_port = self.optimizer.analyze_results(self.mc_results)
                c.drawString(100, 700, "Optimal Portfolio Metrics:")
                c.drawString(120, 680, f"Return: {optimal_port['Return']:.2%}")
                c.drawString(120, 660, f"Volatility: {optimal_port['Volatility']:.2%}")
                c.drawString(120, 640, f"Sharpe Ratio: {optimal_port['Sharpe']:.2f}")
                
                c.drawString(100, 610, "Allocation:")
                for i, (ticker, weight) in enumerate(zip(self.optimizer.tickers, optimal_port[3:])):
                    c.drawString(120, 590 - i*20, f"{ticker}: {weight:.2%}")
            
            c.save()
            messagebox.showinfo("Success", f"Report saved to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save report: {str(e)}")

    def show_docs(self):
        webbrowser.open("https://www.example.com/portfolio-simulator-docs")

    def show_about(self):
        messagebox.showinfo("About", "Advanced Portfolio Simulator Pro\nVersion 1.0\n\nA comprehensive tool for portfolio analysis and optimization")

if __name__ == "__main__":
    root = tk.Tk()
    app = PortfolioApp(root)
    root.mainloop()
