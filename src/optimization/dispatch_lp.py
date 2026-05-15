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