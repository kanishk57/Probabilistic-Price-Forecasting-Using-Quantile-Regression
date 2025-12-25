import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def generate_equity_curve():
    print("Generating Equity Curve...")
    
    # 1. Load Results
    csv_path = Path('results/institutional_backtest.csv')
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run the backtest first.")
        return
        
    df = pd.read_csv(csv_path)
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df = df.sort_values('entry_time')
    
    # 2. Plotting (Matplotlib)
    plt.figure(figsize=(12, 6))
    plt.plot(df['entry_time'], df['capital'], color='#2ecc71', linewidth=2, label='Equity Curve')
    plt.fill_between(df['entry_time'], 10000, df['capital'], color='#2ecc71', alpha=0.1)
    
    plt.title('Institutional Strategy Equity Curve', fontsize=14, pad=15)
    plt.xlabel('Date/Time', fontsize=12)
    plt.ylabel('Account Balance ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Annotate Final ROI
    final_cap = df['capital'].iloc[-1]
    roi = (final_cap - 10000) / 10000 * 100
    plt.annotate(f'Final ROI: {roi:.2f}%', 
                 xy=(df['entry_time'].iloc[-1], final_cap),
                 xytext=(10, 0), textcoords='offset points',
                 fontsize=10, fontweight='bold', color='#27ae60')
    
    plt.tight_layout()
    plt.savefig('results/equity_curve.png', dpi=300)
    print("Saved: results/equity_curve.png")
    
    # 3. Generate HTML Version (Simple Chart.js)
    # We use this for interactive viewing in the browser
    labels = df['entry_time'].dt.strftime('%Y-%m-%d %H:%M').tolist()
    data = df['capital'].tolist()
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Equity Curve - Institutional Strategy</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ font-family: sans-serif; background: #f8f9fa; padding: 20px; }}
            .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Institutional Strategy Equity Curve</h1>
            <canvas id="equityChart"></canvas>
        </div>
        <script>
            const ctx = document.getElementById('equityChart').getContext('2d');
            new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: {labels},
                    datasets: [{{
                        label: 'Account Balance ($)',
                        data: {data},
                        borderColor: '#2ecc71',
                        backgroundColor: 'rgba(46, 204, 113, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.1,
                        pointRadius: 2
                    }}]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        x: {{ title: {{ display: true, text: 'Trade Time' }} }},
                        y: {{ title: {{ display: true, text: 'Balance ($)' }}, min: 9000 }}
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    with open('results/equity_curve.html', 'w') as f:
        f.write(html_content)
    print("Saved: results/equity_curve.html")

if __name__ == "__main__":
    generate_equity_curve()
