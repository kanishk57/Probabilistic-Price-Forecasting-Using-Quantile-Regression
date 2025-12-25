import pandas as pd
import json
from pathlib import Path

def generate_interactive_chart(csv_path, output_path):
    """
    Generate an HTML file with Chart.js to visualize price vs predicted bands.
    """
    df = pd.read_csv(csv_path)
    # Take a sample for visualization to keep HTML size small
    sample = df.tail(200).copy()
    
    # Prepare data for JSON
    times = sample['time'].tolist()
    prices = sample['close'].tolist()
    q10 = sample['q_0.1'].tolist()
    q50 = sample['q_0.5'].tolist()
    q90 = sample['q_0.9'].tolist()
    
    # Calculate price bands (Price * exp(return))
    p_low = (sample['close'] * np.exp(sample['q_0.1'])).tolist()
    p_med = (sample['close'] * np.exp(sample['q_0.5'])).tolist()
    p_high = (sample['close'] * np.exp(sample['q_0.9'])).tolist()

    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Quantile Forecast Visualization</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ font-family: sans-serif; margin: 20px; background: #1a1a1a; color: #eee; }}
            .container {{ width: 90%; margin: auto; }}
            canvas {{ background: #252525; border-radius: 8px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Probabilistic Price Forecasting (1h Horizon)</h1>
            <p>Visualizing 10th, 50th, and 90th percentile confidence bands.</p>
            <canvas id="forecastChart"></canvas>
        </div>
        <script>
            const ctx = document.getElementById('forecastChart').getContext('2d');
            const data = {{
                labels: {json.dumps(times)},
                datasets: [
                    {{
                        label: 'Actual Price',
                        data: {json.dumps(prices)},
                        borderColor: '#ffffff',
                        borderWidth: 2,
                        fill: false,
                        pointRadius: 0
                    }},
                    {{
                        label: 'Upper Bound (90%)',
                        data: {json.dumps(p_high)},
                        borderColor: 'rgba(0, 255, 0, 0.3)',
                        backgroundColor: 'rgba(0, 255, 0, 0.1)',
                        fill: '+1',
                        pointRadius: 0,
                        borderDash: [5, 5]
                    }},
                    {{
                        label: 'Median Forecast (50%)',
                        data: {json.dumps(p_med)},
                        borderColor: '#00ff00',
                        borderWidth: 1,
                        fill: false,
                        pointRadius: 0
                    }},
                    {{
                        label: 'Lower Bound (10%)',
                        data: {json.dumps(p_low)},
                        borderColor: 'rgba(255, 0, 0, 0.3)',
                        backgroundColor: 'rgba(255, 0, 0, 0.1)',
                        fill: false,
                        pointRadius: 0,
                        borderDash: [5, 5]
                    }}
                ]
            }};
            new Chart(ctx, {{
                type: 'line',
                data: data,
                options: {{
                    responsive: true,
                    scales: {{
                        x: {{ ticks: {{ color: '#aaa' }}, grid: {{ color: '#333' }} }},
                        y: {{ ticks: {{ color: '#aaa' }}, grid: {{ color: '#333' }} }}
                    }},
                    plugins: {{
                        legend: {{ labels: {{ color: '#eee' }} }}
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_template)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    import numpy as np
    generate_interactive_chart('results/backtest_trades.csv', 'results/forecast_viz.html')
