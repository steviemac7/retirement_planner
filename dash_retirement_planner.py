# dash_retirement_planner.py
"""
Retirement Planner Dash app with lump sum contributions, adjustable retirement income,
and Monte Carlo simulation for investment returns.

This is a single-file Dash app. Features included:
- Lump-sum contributions (age:amount comma-separated input)
- Year-by-year projection (nominal & inflation-adjusted)
- Adjustable retirement income that can decrease each year starting at an input age
- Monte Carlo simulation (optional) with percentile bands (10-90%, 25-75%) and median
- Charts with x-axis labels showing "Age (Year)" strings
- Hover tooltips showing Age, Year and formatted dollar values (commas)
- CSV download of the numeric projection

Run:
  python dash_retirement_planner.py

Dependencies:
  pip install dash pandas numpy plotly dash-bootstrap-components
"""

import math
import io
import pandas as pd
import numpy as np
from datetime import datetime

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# ----------------------------- Helper functions -----------------------------

def parse_lump_sums(lump_str: str):
    """Parse a text like "45:20000, 50:15000" into {45:20000.0, 50:15000.0}."""
    lump_sums = {}
    if not lump_str:
        return lump_sums
    try:
        pairs = [p.strip() for p in lump_str.split(',') if p.strip()]
        for pair in pairs:
            if ':' not in pair:
                continue
            age_str, amt_str = pair.split(':', 1)
            age = int(age_str.strip())
            amt = float(amt_str.replace('$', '').replace(',', '').strip())
            lump_sums[age] = lump_sums.get(age, 0.0) + amt
    except Exception:
        # on parse error, return what we have
        pass
    return lump_sums


def project_retirement(current_age: int,
                       retirement_age: int,
                       current_savings: float,
                       annual_contribution: float,
                       contribution_growth: float,
                       nominal_return: float,
                       inflation: float,
                       withdrawal_years: int,
                       desired_annual_income: float,
                       social_security: float,
                       lump_sums: dict,
                       income_decrease_enabled: bool = False,
                       income_decrease_start_age: int = None,
                       income_decrease_pct: float = 0.0) -> pd.DataFrame:
    """Return a DataFrame with one row per year of the projection (numeric values).

    Columns: year, age, label, regular_contribution, lump_sum, total_contribution,
    balance_start, returns, withdrawal, balance_end, cumulative_contributions, real_balance, adjusted_income
    """
    years_accum = max(0, retirement_age - current_age)
    total_years = years_accum + withdrawal_years
    rows = []

    balance = float(current_savings)
    cumulative_contrib = 0.0

    for t in range(total_years + 1):
        year = datetime.now().year + t
        age = current_age + t

        # recurring contribution while accumulating
        if t <= years_accum - 1:
            regular_contrib = annual_contribution * ((1 + contribution_growth) ** t)
        else:
            regular_contrib = 0.0

        lump = lump_sums.get(age, 0.0)
        total_contrib = regular_contrib + lump

        balance_start = balance
        returns = balance_start * nominal_return + total_contrib * nominal_return * 0.5
        balance = balance_start + returns + total_contrib

        adjusted_income = 0.0
        withdrawal = 0.0
        if t >= years_accum and withdrawal_years > 0:
            year_in_withdrawal = t - years_accum
            desired_income_nominal = desired_annual_income * ((1 + inflation) ** year_in_withdrawal)

            if income_decrease_enabled and income_decrease_start_age is not None and age >= income_decrease_start_age:
                years_since_start = age - income_decrease_start_age + 1
                adjusted_income = desired_income_nominal * ((1 - income_decrease_pct) ** years_since_start)
            else:
                adjusted_income = desired_income_nominal

            social_nominal = social_security * ((1 + inflation) ** year_in_withdrawal)
            net_withdrawal = max(0.0, adjusted_income - social_nominal)
            withdrawal = min(net_withdrawal, balance)
            balance -= withdrawal

        cumulative_contrib += total_contrib
        real_balance = balance / ((1 + inflation) ** t) if t > 0 else balance

        rows.append({
            "year": year,
            "age": age,
            "label": f"Age {age} ({year})",
            "regular_contribution": round(regular_contrib, 2),
            "lump_sum": round(lump, 2),
            "total_contribution": round(total_contrib, 2),
            "balance_start": round(balance_start, 2),
            "returns": round(returns, 2),
            "withdrawal": round(withdrawal, 2),
            "balance_end": round(balance, 2),
            "cumulative_contributions": round(cumulative_contrib, 2),
            "real_balance": round(real_balance, 2),
            "adjusted_income": round(adjusted_income, 2),
        })

    df = pd.DataFrame(rows)
    return df


def run_monte_carlo(df_base: pd.DataFrame,
                    current_savings: float,
                    annual_contribution: float,
                    contribution_growth: float,
                    nominal_return_mean: float,
                    annual_volatility: float,
                    lump_sums: dict,
                    years_accum: int,
                    withdrawal_years: int,
                    n_sims: int = 1000,
                    seed: int = None):
    """
    Run Monte Carlo simulations. Returns percentile arrays keyed by strings '10','25','50','75','90'.
    df_base is used to read ages and adjusted_income for withdrawals.
    """
    if seed is not None:
        np.random.seed(int(seed))

    total_years = years_accum + withdrawal_years
    sims = np.zeros((n_sims, total_years + 1), dtype=float)

    for sim in range(n_sims):
        bal = float(current_savings)
        sims[sim, 0] = bal
        for t in range(total_years):
            age = int(df_base.iloc[t]['age'])
            if t < years_accum:
                regular_contrib = annual_contribution * ((1 + contribution_growth) ** t)
            else:
                regular_contrib = 0.0
            lump = lump_sums.get(age, 0.0)
            total_contrib = regular_contrib + lump

            r = np.random.normal(loc=nominal_return_mean, scale=annual_volatility)
            returns = bal * r + total_contrib * r * 0.5
            bal = bal + returns + total_contrib

            if t >= years_accum:
                adjusted_income = float(df_base.iloc[t]['adjusted_income'])
                withdrawal = min(max(0.0, adjusted_income), bal)
                bal -= withdrawal

            sims[sim, t + 1] = bal

    pct10 = np.percentile(sims, 10, axis=0)
    pct25 = np.percentile(sims, 25, axis=0)
    pct50 = np.percentile(sims, 50, axis=0)
    pct75 = np.percentile(sims, 75, axis=0)
    pct90 = np.percentile(sims, 90, axis=0)

    # Return with styling suggestions for clearer visualization
    return {
        '10': pct10,
        '25': pct25,
        '50': pct50,
        '75': pct75,
        '90': pct90,
        'raw': sims,
        'style': {
            'bands': [
                {'lower': pct10, 'upper': pct90, 'color': 'rgba(0,100,200,0.5)', 'name': '10-90%'},
                {'lower': pct25, 'upper': pct75, 'color': 'rgba(0,150,100,0.7)', 'name': '25-75%'}
            ],
            'median': {'line': dict(color='red', width=2, dash='solid'), 'name': 'Median'}
        }
    }


# ----------------------------- Dash App Layout ------------------------------

TITLE = "Retirement Planner — Dash (with Monte Carlo)"

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LITERA])
server = app.server

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2(TITLE), width=8),
        dbc.Col(html.Div(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style={"textAlign": "right"}), width=4)
    ], align="center", className="my-3"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Inputs")),
                dbc.CardBody([
                    dbc.Row([dbc.Col(dbc.Label("Current age"), width=6), dbc.Col(dcc.Input(id="current_age", type="number", value=40, min=18, max=100, step=1), width=6)], className="mb-2"),
                    dbc.Row([dbc.Col(dbc.Label("Retirement age"), width=6), dbc.Col(dcc.Input(id="retirement_age", type="number", value=65, min=30, max=100, step=1), width=6)], className="mb-2"),
                    dbc.Row([dbc.Col(dbc.Label("Current savings (CAD)"), width=6), dbc.Col(dcc.Input(id="current_savings", type="number", value=100000, min=0, step=1000), width=6)], className="mb-2"),
                    dbc.Row([dbc.Col(dbc.Label("Annual contribution (today)"), width=6), dbc.Col(dcc.Input(id="annual_contribution", type="number", value=15000, min=0, step=500), width=6)], className="mb-2"),
                    dbc.Row([dbc.Col(dbc.Label("Annual contribution growth (%)"), width=6), dbc.Col(dcc.Input(id="contrib_growth_pct", type="number", value=2.0, min=0, step=0.1), width=6)], className="mb-2"),
                    dbc.Row([dbc.Col(dbc.Label("Expected nominal return (%)"), width=6), dbc.Col(dcc.Input(id="nominal_return_pct", type="number", value=6.0, step=0.1), width=6)], className="mb-2"),
                    dbc.Row([dbc.Col(dbc.Label("Inflation (%)"), width=6), dbc.Col(dcc.Input(id="inflation_pct", type="number", value=2.0, step=0.1), width=6)], className="mb-2"),
                    dbc.Row([dbc.Col(dbc.Label("Years in retirement (estimate)"), width=6), dbc.Col(dcc.Input(id="withdrawal_years", type="number", value=25, min=1, max=60, step=1), width=6)], className="mb-2"),
                    dbc.Row([dbc.Col(dbc.Label("Desired annual retirement income (today's $)"), width=6), dbc.Col(dcc.Input(id="desired_income", type="number", value=50000, min=0, step=1000), width=6)], className="mb-2"),
                    dbc.Row([dbc.Col(dbc.Label("Estimated annual Social Security / Pension (today's $)"), width=6), dbc.Col(dcc.Input(id="social_security", type="number", value=15000, min=0, step=500), width=6)], className="mb-2"),

                    dbc.Row([dbc.Col(dbc.Label("Lump sum contributions (age:amount, comma-separated)"), width=6), dbc.Col(dcc.Input(id="lump_sums", type="text", placeholder="e.g., 45:20000, 50:15000"), width=6)], className="mb-2"),

                    html.Hr(),
                    dbc.Row([dbc.Col(dbc.Label("Enable income decrease (apply during retirement)"), width=8), dbc.Col(dbc.Switch(id='income_decrease_toggle', value=False), width=4)], className="mb-2"),
                    dbc.Row([dbc.Col(dbc.Label("Income decrease start age"), width=6), dbc.Col(dcc.Input(id="income_decrease_start_age", type="number", value=67, min=50, max=120, step=1), width=6)], className="mb-2"),
                    dbc.Row([dbc.Col(dbc.Label("Annual income decrease (%)"), width=6), dbc.Col(dcc.Input(id="income_decrease_pct", type="number", value=5.0, min=0, max=100, step=0.1), width=6)], className="mb-2"),

                    html.Hr(),
                    dbc.Row([dbc.Col(dbc.Label("Monte Carlo: Enable simulations"), width=8), dbc.Col(dbc.Switch(id='mc_toggle', value=False), width=4)], className="mb-2"),
                    dbc.Row([dbc.Col(dbc.Label("Number of simulations"), width=6), dbc.Col(dcc.Input(id="mc_sims", type="number", value=1000, min=100, max=5000, step=100), width=6)], className="mb-2"),
                    dbc.Row([dbc.Col(dbc.Label("Annual volatility (SD %)"), width=6), dbc.Col(dcc.Input(id="mc_vol_pct", type="number", value=12.0, min=0.1, max=100, step=0.1), width=6)], className="mb-2"),
                    dbc.Row([dbc.Col(dbc.Label("Random seed (optional)"), width=6), dbc.Col(dcc.Input(id="mc_seed", type="number", value=42, min=0, step=1), width=6)], className="mb-2"),

                    dbc.Button("Run projection", id="run_button", color="primary", className="mt-2"),
                    html.Span(" "),
                    dbc.Button("Download CSV", id="download_btn", color="secondary", className="mt-2"),
                    dcc.Download(id="download-dataframe-csv"),
                ])
            ])
        ], width=4),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Summary & Charts")),
                dbc.CardBody([
                    html.Div(id="summary_text"),
                    dcc.Graph(id="balance_chart"),
                    dcc.Graph(id="real_balance_chart"),
                    dcc.Graph(id="cashflow_chart"),
                ])
            ])
        ], width=8)
    ]),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H5("Projection Table")),
            dbc.CardBody([
                dcc.Loading(dcc.Graph(id="table_graph")),
            ])
        ]))
    ], className="my-3")

], fluid=True)

# ----------------------------- Callbacks -----------------------------------

@app.callback(
    Output('summary_text', 'children'),
    Output('balance_chart', 'figure'),
    Output('real_balance_chart', 'figure'),
    Output('cashflow_chart', 'figure'),
    Output('table_graph', 'figure'),
    Input('run_button', 'n_clicks'),
    State('current_age', 'value'),
    State('retirement_age', 'value'),
    State('current_savings', 'value'),
    State('annual_contribution', 'value'),
    State('contrib_growth_pct', 'value'),
    State('nominal_return_pct', 'value'),
    State('inflation_pct', 'value'),
    State('withdrawal_years', 'value'),
    State('desired_income', 'value'),
    State('social_security', 'value'),
    State('lump_sums', 'value'),
    State('income_decrease_toggle', 'value'),
    State('income_decrease_start_age', 'value'),
    State('income_decrease_pct', 'value'),
    State('mc_toggle', 'value'),
    State('mc_sims', 'value'),
    State('mc_vol_pct', 'value'),
    State('mc_seed', 'value'),
)
def update_projections(n_clicks, current_age, retirement_age, current_savings,
                       annual_contribution, contrib_growth_pct, nominal_return_pct,
                       inflation_pct, withdrawal_years, desired_income, social_security,
                       lump_sums_str, income_decrease_toggle, income_decrease_start_age,
                       income_decrease_pct, mc_toggle, mc_sims, mc_vol_pct, mc_seed):
    try:
        contrib_growth = float(contrib_growth_pct) / 100.0
    except Exception:
        contrib_growth = 0.02
    try:
        nominal_return = float(nominal_return_pct) / 100.0
    except Exception:
        nominal_return = 0.06
    try:
        inflation = float(inflation_pct) / 100.0
    except Exception:
        inflation = 0.02

    income_decrease_enabled = bool(income_decrease_toggle)
    income_decrease_start_age = int(income_decrease_start_age) if income_decrease_start_age is not None else None
    try:
        income_decrease_rate = float(income_decrease_pct) / 100.0
    except Exception:
        income_decrease_rate = 0.0

    lump_sums = parse_lump_sums(lump_sums_str)

    df = project_retirement(int(current_age), int(retirement_age), float(current_savings),
                            float(annual_contribution), contrib_growth, nominal_return, inflation,
                            int(withdrawal_years), float(desired_income), float(social_security),
                            lump_sums, income_decrease_enabled, income_decrease_start_age, income_decrease_rate)

    final_row = df.iloc[-1]
    summary = html.Div([
        html.P(f"Projection horizon: {df['age'].iloc[0]} → {df['age'].iloc[-1]} ({len(df)-1} years simulated)"),
        html.P(f"Final nominal balance: ${final_row['balance_end']:,.0f}"),
        html.P(f"Final real (today\'s $) balance: ${final_row['real_balance']:,.0f}"),
        html.P(f"Total contributions (nominal): ${df['cumulative_contributions'].iloc[-1]:,.0f}"),
        html.P(f"Total withdrawals taken in retirement: ${df['withdrawal'].sum():,.0f}"),
    ])

    # x-axis labels (Age (Year))
    x = df['label']

    # Balance chart
    fig_balance = go.Figure()

    if mc_toggle:
        try:
            sims = int(mc_sims)
            vol = float(mc_vol_pct) / 100.0
            seed = int(mc_seed) if mc_seed is not None else None
            years_accum = max(0, int(retirement_age) - int(current_age))
            mc = run_monte_carlo(df, float(current_savings), float(annual_contribution), contrib_growth, nominal_return, vol, lump_sums, years_accum, int(withdrawal_years), n_sims=sims, seed=seed)

            # 10-90 band
            fig_balance.add_trace(go.Scatter(x=x, y=mc['90'], line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig_balance.add_trace(go.Scatter(x=x, y=mc['10'], fill='tonexty', fillcolor='rgba(255,148,112,0.6)', line=dict(width=0), name='10–90% band', hoverinfo='skip'))
            # 25-75 band
            fig_balance.add_trace(go.Scatter(x=x, y=mc['75'], line=dict(width=0), showlegend=False, hoverinfo='skip'))
            fig_balance.add_trace(go.Scatter(x=x, y=mc['25'], fill='tonexty', fillcolor='rgba(144,238,144,0.6)', line=dict(width=0), name='25–75% band', hoverinfo='skip'))
            # median
            fig_balance.add_trace(go.Scatter(x=x, y=mc['50'], mode='lines', line=dict(color='blue', dash='dash'), name='MC Median'))
        except Exception as e:
            print('Monte Carlo failed:', e)

    # deterministic lines
    fig_balance.add_trace(go.Scatter(x=x, y=df['balance_end'], mode='lines+markers', name='Nominal Balance',
                                     hovertemplate='%{x}<br>Balance: $%{y:,.0f}<extra></extra>'))
    fig_balance.add_trace(go.Scatter(x=x, y=df['cumulative_contributions'], mode='lines', name='Cumulative Contributions',
                                     hovertemplate='%{x}<br>Cumulative Contributions: $%{y:,.0f}<extra></extra>'))

    # Lump sum markers (with formatted customdata)
    lump_df = df[df['lump_sum'] > 0]
    if not lump_df.empty:
        formatted_amounts = [f"${v:,.0f}" for v in lump_df['lump_sum']]
        customdata = [[fa, int(a)] for fa, a in zip(formatted_amounts, lump_df['age'])]
        fig_balance.add_trace(go.Scatter(
            x=lump_df['label'],
            y=lump_df['balance_end'],
            mode='markers+text',
            text=formatted_amounts,
            textposition='top center',
            marker=dict(color='red', size=10, symbol='star'),
            name='Lump Sum Contribution',
            hovertemplate='Lump Sum: %{customdata[0]}<br>Age: %{customdata[1]}<extra></extra>',
            customdata=customdata
        ))

    # Adjusted income line on secondary y-axis
    fig_balance.add_trace(go.Scatter(x=x, y=df['adjusted_income'], mode='lines', name='Adjusted Retirement Income', line=dict(dash='dash', width=2), yaxis='y2',
                                     hovertemplate='%{x}<br>Adjusted Income: $%{y:,.0f}<extra></extra>'))

    fig_balance.update_layout(title='Nominal Portfolio Balance', xaxis_title='Age (Year)',
                              yaxis=dict(title='CAD (Balance)', side='left', showgrid=True),
                              yaxis2=dict(title='CAD (Adjusted Income)', overlaying='y', side='right'),
                              legend=dict(x=0.01, y=0.99))

    # Real balance chart
    fig_real = go.Figure()
    fig_real.add_trace(go.Scatter(x=x, y=df['real_balance'], mode='lines+markers', name='Real Balance (today$)',
                                  hovertemplate='%{x}<br>Real Balance (today$): $%{y:,.0f}<extra></extra>'))
    fig_real.update_layout(title="Inflation-adjusted Balance (Today's dollars)", xaxis_title='Age (Year)', yaxis_title="Today's CAD")

    # Cashflow chart: regular contributions and lump sums stacked; withdrawals shown downward
    fig_cash = go.Figure()
    reg_fmt = [f"${v:,.2f}" for v in df['regular_contribution']]
    lump_fmt = [f"${v:,.2f}" for v in df['lump_sum']]
    wd_fmt = [f"${v:,.2f}" for v in df['withdrawal']]

    fig_cash.add_trace(go.Bar(x=x, y=df['regular_contribution'], name='Regular Contribution', customdata=reg_fmt,
                              hovertemplate='%{x}<br>Regular Contribution: %{customdata}<extra></extra>'))
    fig_cash.add_trace(go.Bar(x=x, y=df['lump_sum'], name='Lump Sum', customdata=lump_fmt,
                              hovertemplate='%{x}<br>Lump Sum: %{customdata}<extra></extra>'))
    fig_cash.add_trace(go.Bar(x=x, y=-df['withdrawal'], name='Withdrawal', customdata=wd_fmt,
                              hovertemplate='%{x}<br>Withdrawal: %{customdata}<extra></extra>'))
    fig_cash.update_layout(barmode='stack', title='Annual Contributions (stacked regular + lump) & Withdrawals (downwards)', xaxis_title='Age (Year)', yaxis_title='CAD')

    # Projection table (formatted strings)
    display_cols = ['year', 'age', 'regular_contribution', 'lump_sum', 'total_contribution', 'withdrawal', 'balance_end', 'real_balance', 'adjusted_income']
    df_display = df.copy()
    for c in ['regular_contribution', 'lump_sum', 'total_contribution', 'balance_start', 'returns', 'withdrawal', 'balance_end', 'cumulative_contributions', 'real_balance', 'adjusted_income']:
        if c in df_display.columns:
            df_display[c] = df_display[c].apply(lambda v: f"{v:,.2f}")

    fig_table = go.Figure(data=[go.Table(
        header=dict(values=[c.replace('_', ' ').title() for c in ['Year', 'Age'] + display_cols[2:]], fill_color='paleturquoise', align='left'),
        cells=dict(values=[df_display['year'], df_display['age']] + [df_display[c] for c in display_cols[2:]], fill_color='lavender', align='left')
    )])
    fig_table.update_layout(height=500)

    return summary, fig_balance, fig_real, fig_cash, fig_table


# CSV download callback
@app.callback(
    Output('download-dataframe-csv', 'data'),
    Input('download_btn', 'n_clicks'),
    State('current_age', 'value'),
    State('retirement_age', 'value'),
    State('current_savings', 'value'),
    State('annual_contribution', 'value'),
    State('contrib_growth_pct', 'value'),
    State('nominal_return_pct', 'value'),
    State('inflation_pct', 'value'),
    State('withdrawal_years', 'value'),
    State('desired_income', 'value'),
    State('social_security', 'value'),
    State('lump_sums', 'value'),
    State('income_decrease_toggle', 'value'),
    State('income_decrease_start_age', 'value'),
    State('income_decrease_pct', 'value'),
    prevent_initial_call=True
)
def download_csv(n_clicks, current_age, retirement_age, current_savings,
                 annual_contribution, contrib_growth_pct, nominal_return_pct,
                 inflation_pct, withdrawal_years, desired_income, social_security,
                 lump_sums_str, income_decrease_toggle, income_decrease_start_age,
                 income_decrease_pct):
    contrib_growth = float(contrib_growth_pct) / 100.0
    nominal_return = float(nominal_return_pct) / 100.0
    inflation = float(inflation_pct) / 100.0
    income_decrease_enabled = bool(income_decrease_toggle)
    income_decrease_start_age = int(income_decrease_start_age) if income_decrease_start_age is not None else None
    income_decrease_rate = float(income_decrease_pct) / 100.0
    lump_sums = parse_lump_sums(lump_sums_str)

    df = project_retirement(int(current_age), int(retirement_age), float(current_savings),
                            float(annual_contribution), contrib_growth, nominal_return, inflation,
                            int(withdrawal_years), float(desired_income), float(social_security),
                            lump_sums, income_decrease_enabled, income_decrease_start_age, income_decrease_rate)

    export_cols = ['year', 'age', 'regular_contribution', 'lump_sum', 'total_contribution', 'balance_start', 'returns', 'withdrawal', 'balance_end', 'cumulative_contributions', 'real_balance', 'adjusted_income']
    return dcc.send_data_frame(df[export_cols].to_csv, f"retirement_projection_{int(current_age)}_{int(retirement_age)}.csv", index=False)


# ----------------------------- Run server ----------------------------------

if __name__ == '__main__':
    app.run(debug=True)
