import numpy as np
import pandas as pd
import pulp
import matplotlib.pyplot as plt

def run_degradation_backtest(df_val_results, degradation_cost_per_mwh=5.00):
    """
    Executes a daily walk-forward backtest incorporating a physical battery cell 
    degradation penalty within the Stochastic MILP solver framework.
    
    Parameters:
    -----------
    df_val_results : pd.DataFrame
        Continuous hourly data containing actual and quantile forecasted prices.
    degradation_cost_per_mwh : float
        Virtual penalty fee (EUR/MWh) applied to charging and discharging commands
        to protect hardware assets from low-margin cycling.
    """
    n_days = len(df_val_results) // 24
    df_clean = df_val_results.iloc[:n_days * 24].copy()
    
    backtest_records = []
    
    # Define physical battery attributes
    POWER_CAPACITY_MW = 10.0
    ENERGY_CAPACITY_MWh = 20.0
    EFFICIENCY = 0.85
    INITIAL_SOC = 0.20
    MIN_SOC = 0.10
    MAX_SOC = 0.90
    
    print(f"Backtesting {n_days} days using Degradation-Aware Stochastic Optimization...")
    print(f"🔧 Cell Degradation Penalty Set To: {degradation_cost_per_mwh:.2f} EUR/MWh cycled\n")
    
    for day in range(n_days):
        day_data = df_clean.iloc[day*24 : (day+1)*24]
        
        actual_prices = day_data['actual_price'].values
        q05_prices = day_data['pred_price_q05'].values
        q50_prices = day_data['pred_price_q50'].values
        q95_prices = day_data['pred_price_q95'].values
        
        # Initialize optimization structure
        prob = pulp.LpProblem(f"Degradation_BESS_Day_{day}", pulp.LpMaximize)
        T = range(24)
        
        # Decision variables
        p_charge = pulp.LpVariable.dicts("Charge", T, lowBound=0, upBound=POWER_CAPACITY_MW)
        p_discharge = pulp.LpVariable.dicts("Discharge", T, lowBound=0, upBound=POWER_CAPACITY_MW)
        soc = pulp.LpVariable.dicts("SoC", T, lowBound=MIN_SOC * ENERGY_CAPACITY_MWh, 
                                                upBound=MAX_SOC * ENERGY_CAPACITY_MWh)
        is_charging = pulp.LpVariable.dicts("is_charging", T, cat=pulp.LpBinary)
        is_discharging = pulp.LpVariable.dicts("is_discharging", T, cat=pulp.LpBinary)
        
        # Physical parameter constraints
        for t in T:
            prob += p_charge[t] <= is_charging[t] * POWER_CAPACITY_MW
            prob += p_discharge[t] <= is_discharging[t] * POWER_CAPACITY_MW
            prob += is_charging[t] + is_discharging[t] <= 1
            
            if t == 0:
                prob += soc[t] == (INITIAL_SOC * ENERGY_CAPACITY_MWh) + \
                                  (p_charge[t] * np.sqrt(EFFICIENCY)) - \
                                  (p_discharge[t] / np.sqrt(EFFICIENCY))
            else:
                prob += soc[t] == soc[t-1] + \
                                  (p_charge[t] * np.sqrt(EFFICIENCY)) - \
                                  (p_discharge[t] / np.sqrt(EFFICIENCY))
                
        # FIX: Target only the final hour of the day (Hour 23) to complete the daily loop
        prob += soc[23] == (INITIAL_SOC * ENERGY_CAPACITY_MWh)

        # Upgraded Objective Function: Optimizes expected returns across quantiles, 
        # minus the direct physical cost of cell material degradation.
        prob += pulp.lpSum([
            0.20 * (p_discharge[t] * q05_prices[t] - p_charge[t] * q05_prices[t]) +
            0.60 * (p_discharge[t] * q50_prices[t] - p_charge[t] * q50_prices[t]) +
            0.20 * (p_discharge[t] * q95_prices[t] - p_charge[t] * q95_prices[t]) -
            (degradation_cost_per_mwh * (p_charge[t] + p_discharge[t]))
            for t in T
        ])
        
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        # Post-optim accounting logs (Realized returns discard the virtual hardware penalty)
        for t in T:
            c_cmd = p_charge[t].varValue if p_charge[t].varValue is not None else 0.0
            d_cmd = p_discharge[t].varValue if p_discharge[t].varValue is not None else 0.0
            
            expected_profit_step = (d_cmd * q50_prices[t]) - (c_cmd * q50_prices[t])
            actual_profit_step = (d_cmd * actual_prices[t]) - (c_cmd * actual_prices[t])
            
            backtest_records.append({
                "Day": day,
                "Hour": t,
                "Actual_Price_EUR": actual_prices[t],
                "Pred_Price_Median_EUR": q50_prices[t],
                "Charge_Command_MW": c_cmd,
                "Discharge_Command_MW": d_cmd,
                "End_SoC_MWh": soc[t].varValue,
                "Expected_Profit_EUR": expected_profit_step,
                "Actual_Realized_Profit_EUR": actual_profit_step,
                "MWh_Cycled": (c_cmd + d_cmd)
            })

    df_backtest = pd.DataFrame(backtest_records)
    
    total_expected = df_backtest["Expected_Profit_EUR"].sum()
    total_realized = df_backtest["Actual_Realized_Profit_EUR"].sum()
    total_mwh_cycled = df_backtest["MWh_Cycled"].sum()
    realization_ratio = (total_realized / total_expected) * 100 if total_expected > 0 else 0
    
    print("Backtest Execution Complete!")
    print(f"Total Expected Revenue: {total_expected:,.2f} EUR")
    print(f"Total Realized Revenue: {total_realized:,.2f} EUR")
    print(f"Profit Realization Ratio: {realization_ratio:.2f}%")
    print(f"Total Physical Energy Cycled: {total_mwh_cycled:,.2f} MWh")
    
    return df_backtest


def plot_backtest_day(df_ledger, target_day=0):
    """
    Generates a dual-axis operational profile plot for an asset dispatch day.
    """
    day_df = df_ledger[df_ledger["Day"] == target_day].copy()
    
    fig, ax1 = plt.subplots(figsize=(14, 7), dpi=100)
    
    # Left Axis Mapping: Market Prices
    color_price = '#1f77b4'
    ax1.set_xlabel('Trading Hour (Day-Ahead Horizon)', fontsize=12, labelpad=10)
    ax1.set_ylabel('Market Spot Price (EUR/MWh)', color=color_price, fontsize=12)
    
    # Plot real market clearing values vs model expectations
    ax1.plot(day_df['Hour'], day_df['Actual_Price_EUR'], color=color_price, linewidth=2.5, label='Actual Spot Price')
    ax1.plot(day_df['Hour'], day_df['Pred_Price_Median_EUR'], color=color_price, linestyle='--', alpha=0.5, label='LSTM Expected Median')
    ax1.tick_params(axis='y', labelcolor=color_price)
    ax1.grid(True, alpha=0.2, linestyle=':')
    
    # Overlap visual bars for real-time dispatch decisions
    ax1.bar(day_df['Hour'], day_df['Charge_Command_MW'], color='#d62728', alpha=0.25, width=0.6, label='Charging Action (Buy/Inject)')
    ax1.bar(day_df['Hour'], -day_df['Discharge_Command_MW'], color='#2ca02c', alpha=0.25, width=0.6, label='Discharging Action (Sell/Eject)')
    
    # Right Axis Mapping: Battery Cell Capacity State (SoC)
    ax2 = ax1.twinx()  
    color_soc = '#2ca02c'
    ax2.set_ylabel('Battery State of Charge (SoC MWh)', color=color_soc, fontsize=12, labelpad=10)
    ax2.step(day_df['Hour'], day_df['End_SoC_MWh'], color=color_soc, linewidth=2.0, where='mid', label='Asset SoC Level')
    ax2.tick_params(axis='y', labelcolor=color_soc)
    
    # Consolidate and format plot legends cleanly
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', frameon=True, facecolor='white', edgecolor='none')
    
    # Print localized performance metadata directly onto the plot title
    day_profit = day_df["Actual_Realized_Profit_EUR"].sum()
    day_cycling = day_df["MWh_Cycled"].sum()
    plt.title(f'BESS Dispatch Strategy & Market Valuation | Day {target_day} (Realized Return: +{day_profit:,.2f} EUR | Throughput: {day_cycling:.1f} MWh)', 
              fontsize=13, fontweight='bold', pad=15)
    
    fig.tight_layout()
    plt.show()
import pandas as pd
from pulp import *
from src.models.predict import predict_next_days

def optimize_dispatch(forecast_df: pd.DataFrame, 
                     pumped_storage_capacity: float = 500,   # MW
                     horizon_hours: int = 24):
    """
    Linear Programming Dispatch Optimization
    """
    prob = LpProblem("Energy_Dispatch_Optimization", LpMinimize)
    
    # Time periods (15-min)
    T = range(len(forecast_df))
    
    # Variables
    gen_gas = LpVariable.dicts("Gas", T, lowBound=0)
    gen_hard_coal = LpVariable.dicts("Coal", T, lowBound=0)
    pumped_charge = LpVariable.dicts("Pumped_Charge", T, lowBound=0)
    pumped_discharge = LpVariable.dicts("Pumped_Discharge", T, lowBound=0)
    
    # Objective: Minimize cost (example prices)
    prob += lpSum([
        gen_gas[t] * 80 +          # €/MWh
        gen_hard_coal[t] * 65 + 
        pumped_charge[t] * 5       # efficiency loss cost
        for t in T
    ])
    
    # Constraints
    for t in T:
        # Supply = Demand
        prob += (
            forecast_df['predicted_gen_solar'].iloc[t] + 
            forecast_df['predicted_gen_onshore_wind'].iloc[t] + 
            gen_gas[t] + gen_hard_coal[t] + 
            pumped_discharge[t] - pumped_charge[t] 
            >= forecast_df['predicted_cons_total_grid'].iloc[t]
        )
        
        # Pumped storage dynamics (simplified)
        if t > 0:
            prob += pumped_storage_level[t] == pumped_storage_level[t-1] + \
                    pumped_charge[t-1]*0.9 - pumped_discharge[t-1]
    
    # Solve
    prob.solve(PULP_CBC_CMD(msg=0))
    
    print("Optimization Status:", LpStatus[prob.status])
    print(f"Total Expected Cost: €{value(prob.objective):,.2f}")
    
    # Extract results
    results = pd.DataFrame({
        'timestamp': forecast_df['timestamp'],
        'gas_gen': [value(gen_gas[t]) for t in T],
        'coal_gen': [value(gen_hard_coal[t]) for t in T],
        'pumped_charge': [value(pumped_charge[t]) for t in T],
        'pumped_discharge': [value(pumped_discharge[t]) for t in T]
    })
    
    return results