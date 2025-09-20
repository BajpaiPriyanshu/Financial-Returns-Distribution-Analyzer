import pandas as pd
import yfinance as yf
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class FinancialReturnsAnalyzer:
    def __init__(self, symbols, period='2y'):
        self.symbols = symbols
        self.period = period
        self.data = {}
        self.returns = {}
        self.distribution_fits = {}
        
    def fetch_data(self):
        print("Fetching financial data...")
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=self.period)
                self.data[symbol] = hist['Close']
                print(f"Successfully fetched data for {symbol}")
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
    
    def calculate_returns(self):
        print("\nCalculating returns...")
        for symbol in self.symbols:
            if symbol in self.data:
                prices = self.data[symbol]
                returns = prices.pct_change().dropna()
                self.returns[symbol] = returns
                print(f"Calculated {len(returns)} returns for {symbol}")
    
    def calculate_moments(self, data):
        return {
            'mean': np.mean(data),
            'variance': np.var(data, ddof=1),
            'std': np.std(data, ddof=1),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data, fisher=True)
        }
    
    def fit_normal_distribution(self, data):
        mu, sigma = stats.norm.fit(data)
        ll = np.sum(stats.norm.logpdf(data, mu, sigma))
        n = len(data)
        k = 2
        aic = 2*k - 2*ll
        bic = k*np.log(n) - 2*ll
        return {
            'distribution': 'Normal',
            'parameters': {'mu': mu, 'sigma': sigma},
            'log_likelihood': ll,
            'aic': aic,
            'bic': bic
        }
    
    def fit_t_distribution(self, data):
        df, mu, sigma = stats.t.fit(data)
        ll = np.sum(stats.t.logpdf(data, df, mu, sigma))
        n = len(data)
        k = 3
        aic = 2*k - 2*ll
        bic = k*np.log(n) - 2*ll
        return {
            'distribution': 'Student-t',
            'parameters': {'df': df, 'mu': mu, 'sigma': sigma},
            'log_likelihood': ll,
            'aic': aic,
            'bic': bic
        }
    
    def fit_skewed_normal_distribution(self, data):
        a, mu, sigma = stats.skewnorm.fit(data)
        ll = np.sum(stats.skewnorm.logpdf(data, a, mu, sigma))
        n = len(data)
        k = 3
        aic = 2*k - 2*ll
        bic = k*np.log(n) - 2*ll
        return {
            'distribution': 'Skewed Normal',
            'parameters': {'a': a, 'mu': mu, 'sigma': sigma},
            'log_likelihood': ll,
            'aic': aic,
            'bic': bic
        }
    
    def perform_goodness_of_fit_tests(self, data, fit_results):
        results = {}
        for fit in fit_results:
            dist_name = fit['distribution']
            params = fit['parameters']
            if dist_name == 'Normal':
                ks_stat, ks_p = stats.kstest(data, lambda x: stats.norm.cdf(x, params['mu'], params['sigma']))
            elif dist_name == 'Student-t':
                ks_stat, ks_p = stats.kstest(data, lambda x: stats.t.cdf(x, params['df'], params['mu'], params['sigma']))
            elif dist_name == 'Skewed Normal':
                ks_stat, ks_p = stats.kstest(data, lambda x: stats.skewnorm.cdf(x, params['a'], params['mu'], params['sigma']))
            results[dist_name] = {
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p
            }
        return results
    
    def fit_distributions(self):
        print("\nFitting distributions...")
        for symbol in self.returns:
            data = self.returns[symbol].values
            normal_fit = self.fit_normal_distribution(data)
            t_fit = self.fit_t_distribution(data)
            skewed_fit = self.fit_skewed_normal_distribution(data)
            fits = [normal_fit, t_fit, skewed_fit]
            gof_tests = self.perform_goodness_of_fit_tests(data, fits)
            moments = self.calculate_moments(data)
            self.distribution_fits[symbol] = {
                'fits': fits,
                'gof_tests': gof_tests,
                'moments': moments,
                'data': data
            }
            print(f"Fitted distributions for {symbol}")
    
    def create_comparison_table(self):
        comparison_data = []
        for symbol in self.distribution_fits:
            fits = self.distribution_fits[symbol]['fits']
            moments = self.distribution_fits[symbol]['moments']
            for fit in fits:
                row = {
                    'Symbol': symbol,
                    'Distribution': fit['distribution'],
                    'Log-Likelihood': fit['log_likelihood'],
                    'AIC': fit['aic'],
                    'BIC': fit['bic'],
                    'Mean': moments['mean'],
                    'Std Dev': moments['std'],
                    'Skewness': moments['skewness'],
                    'Kurtosis': moments['kurtosis']
                }
                comparison_data.append(row)
        return pd.DataFrame(comparison_data)
    
    def plot_distribution_fits(self):
        n_symbols = len(self.returns)
        n_figures = (n_symbols + 2) // 3
        for fig_idx in range(n_figures):
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'Distribution Fits Comparison - Figure {fig_idx + 1}', fontsize=16, fontweight='bold')
            start_idx = fig_idx * 3
            end_idx = min((fig_idx + 1) * 3, n_symbols)
            symbols_subset = list(self.returns.keys())[start_idx:end_idx]
            for i, symbol in enumerate(symbols_subset):
                ax = axes[i] if len(symbols_subset) > 1 else axes
                data = self.distribution_fits[symbol]['data']
                fits = self.distribution_fits[symbol]['fits']
                ax.hist(data, bins=50, density=True, alpha=0.7, color='lightblue', 
                       edgecolor='black', label='Actual Returns')
                x = np.linspace(data.min(), data.max(), 100)
                colors = ['red', 'green', 'purple']
                for j, fit in enumerate(fits):
                    params = fit['parameters']
                    if fit['distribution'] == 'Normal':
                        y = stats.norm.pdf(x, params['mu'], params['sigma'])
                    elif fit['distribution'] == 'Student-t':
                        y = stats.t.pdf(x, params['df'], params['mu'], params['sigma'])
                    elif fit['distribution'] == 'Skewed Normal':
                        y = stats.skewnorm.pdf(x, params['a'], params['mu'], params['sigma'])
                    ax.plot(x, y, color=colors[j], linewidth=2, 
                           label=f"{fit['distribution']} (AIC: {fit['aic']:.2f})")
                ax.set_title(f'{symbol} Returns Distribution', fontweight='bold')
                ax.set_xlabel('Returns')
                ax.set_ylabel('Density')
                ax.legend()
                ax.grid(True, alpha=0.3)
            for i in range(len(symbols_subset), 3):
                if len(symbols_subset) > 1:
                    axes[i].set_visible(False)
            plt.tight_layout()
            plt.show()
    
    def plot_qq_plots(self):
        n_symbols = len(self.returns)
        n_figures = (n_symbols + 2) // 3
        for fig_idx in range(n_figures):
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'Q-Q Plots - Figure {fig_idx + 1}', fontsize=16, fontweight='bold')
            start_idx = fig_idx * 3
            end_idx = min((fig_idx + 1) * 3, n_symbols)
            symbols_subset = list(self.returns.keys())[start_idx:end_idx]
            for i, symbol in enumerate(symbols_subset):
                ax = axes[i] if len(symbols_subset) > 1 else axes
                data = self.distribution_fits[symbol]['data']
                fits = self.distribution_fits[symbol]['fits']
                sorted_data = np.sort(data)
                n = len(sorted_data)
                p = np.arange(1, n + 1) / (n + 1)
                best_fit = min(fits, key=lambda x: x['aic'])
                params = best_fit['parameters']
                if best_fit['distribution'] == 'Normal':
                    theoretical_q = stats.norm.ppf(p, params['mu'], params['sigma'])
                elif best_fit['distribution'] == 'Student-t':
                    theoretical_q = stats.t.ppf(p, params['df'], params['mu'], params['sigma'])
                elif best_fit['distribution'] == 'Skewed Normal':
                    theoretical_q = stats.skewnorm.ppf(p, params['a'], params['mu'], params['sigma'])
                ax.scatter(theoretical_q, sorted_data, alpha=0.6, s=20)
                ax.plot([theoretical_q.min(), theoretical_q.max()], 
                       [theoretical_q.min(), theoretical_q.max()], 
                       'r--', linewidth=2, label='Perfect Fit')
                ax.set_title(f'{symbol} - {best_fit["distribution"]} Q-Q Plot', fontweight='bold')
                ax.set_xlabel('Theoretical Quantiles')
                ax.set_ylabel('Sample Quantiles')
                ax.legend()
                ax.grid(True, alpha=0.3)
            for i in range(len(symbols_subset), 3):
                if len(symbols_subset) > 1:
                    axes[i].set_visible(False)
            plt.tight_layout()
            plt.show()
    
    def plot_moments_comparison(self):
        symbols = list(self.distribution_fits.keys())
        moments_data = []
        for symbol in symbols:
            moments = self.distribution_fits[symbol]['moments']
            moments_data.append({
                'Symbol': symbol,
                'Mean': moments['mean'],
                'Std Dev': moments['std'],
                'Skewness': moments['skewness'],
                'Kurtosis': moments['kurtosis']
            })
        df_moments = pd.DataFrame(moments_data)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Statistical Moments Comparison Across Assets', fontsize=16, fontweight='bold')
        axes[0].scatter(df_moments['Mean'], df_moments['Std Dev'], 
                       s=100, alpha=0.7, c='blue')
        for i, symbol in enumerate(df_moments['Symbol']):
            axes[0].annotate(symbol, (df_moments['Mean'][i], df_moments['Std Dev'][i]),
                           xytext=(5, 5), textcoords='offset points')
        axes[0].set_xlabel('Mean Return')
        axes[0].set_ylabel('Standard Deviation')
        axes[0].set_title('Risk-Return Profile', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[1].bar(df_moments['Symbol'], df_moments['Skewness'], 
                   color='green', alpha=0.7)
        axes[1].set_ylabel('Skewness')
        axes[1].set_title('Distribution Skewness', fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[2].bar(df_moments['Symbol'], df_moments['Kurtosis'], 
                   color='purple', alpha=0.7)
        axes[2].set_ylabel('Excess Kurtosis')
        axes[2].set_title('Distribution Kurtosis', fontweight='bold')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
    
    def generate_report(self):
        print("\n" + "="*60)
        print("FINANCIAL RETURNS DISTRIBUTION ANALYSIS REPORT")
        print("="*60)
        for symbol in self.distribution_fits:
            print(f"\n{symbol.upper()} ANALYSIS:")
            print("-" * 30)
            fits = self.distribution_fits[symbol]['fits']
            moments = self.distribution_fits[symbol]['moments']
            gof_tests = self.distribution_fits[symbol]['gof_tests']
            best_fit = min(fits, key=lambda x: x['aic'])
            print(f"Best Fit Distribution: {best_fit['distribution']}")
            print(f"Parameters: {best_fit['parameters']}")
            print(f"AIC: {best_fit['aic']:.4f}")
            print(f"BIC: {best_fit['bic']:.4f}")
            print("\nStatistical Moments:")
            for moment, value in moments.items():
                print(f"  {moment.capitalize()}: {value:.6f}")
            print("\nGoodness-of-Fit Tests (K-S Test p-values):")
            for dist, test_results in gof_tests.items():
                significance = "Good fit" if test_results['ks_p_value'] > 0.05 else "Poor fit"
                print(f"  {dist}: {test_results['ks_p_value']:.4f} ({significance})")
        print("\n" + "="*60)
        comparison_df = self.create_comparison_table()
        print("\nDETAILED COMPARISON TABLE:")
        print(comparison_df.round(6))
        return comparison_df
    
    def run_complete_analysis(self):
        print("Starting Financial Returns Distribution Analysis...")
        self.fetch_data()
        self.calculate_returns()
        self.fit_distributions()
        self.plot_distribution_fits()
        self.plot_qq_plots()
        self.plot_moments_comparison()
        comparison_df = self.generate_report()
        return comparison_df

if __name__ == "__main__":
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'SPY', 'BTC-USD']
    analyzer = FinancialReturnsAnalyzer(symbols, period='2y')
    results = analyzer.run_complete_analysis()
    print("\nAnalysis completed successfully!")
    print("The results DataFrame contains detailed comparison metrics.")
    print("Visualizations show distribution fits, Q-Q plots, and moments comparison.")
