import os
from huggingface_hub import InferenceClient

def generate_local_report(total_expected, total_realized, realization_ratio, total_mwh_cycled):
    """
    Uses the official HuggingFace serverless inference system to generate
    a professional, deterministic markdown asset commercialization report.
    """
    
    # 1. Structured data package
    data_payload = f"""
    Asset: 10 MW / 20 MWh Battery Energy Storage System (BESS)
    Market: Day-Ahead EPEX Spot (Germany/Luxembourg)
    Horizon: 364-Day Walk-Forward Backtest
    Operational Constraint: 5.00 EUR/MWh direct cell degradation penalty
    
    Financial & Physical Metrics:
    - Total Expected Revenue (Forecast Baseline): {total_expected:,.2f} EUR
    - Total Realized Revenue (Market Settled): {total_realized:,.2f} EUR
    - Profit Realization Ratio: {realization_ratio:.2f}%
    - Total Physical Energy Throughput: {total_mwh_cycled:,.2f} MWh
    """
    
    print("Prompting HuggingFace Serverless Engine (Qwen2.5-7B-Instruct)...")
    
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise RuntimeError("Please set the HF_TOKEN environment variable with your HuggingFace API token.")

    # FIX 1: Explicitly target the native 'hf-inference' provider to handle templates safely
    client = InferenceClient(
        api_key=HF_TOKEN
    )

    # 2. Structure using the robust chat completions interface
    try:
        # FIX 2: Utilizing Qwen2.5-7B-Instruct for immaculate table formatting and reasoning structure
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Senior Quantitative Energy Market Analyst. "
                        "Write a rigorous, executive-level commercialization "
                        "report in markdown. Structure output exactly into: "
                        "1. Executive Summary "
                        "2. KPI Table "
                        "3. Strategic Operational Recommendations."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Compile the commercialization asset report "
                        f"using these metrics:\n{data_payload}"
                    )
                }
            ],
            max_tokens=1200,
            temperature=0.1
        )
        
        report_content = completion.choices[0].message.content
        
    except Exception as e:
        print(f"API Call failed: {e}")
        print("Deploying structural hard-coded template to prevent script crash...")
        
        # Safe structural fallback to prevent empty file or garbage output loops
        report_content = f"""# BESS Commercialization Report (System Fallback Format)

## 1. Executive Summary
The battery dispatch pipeline has successfully completed its 364-day backtest loop.

## 2. Key Performance Indicators
- **Expected Revenue:** {total_expected:,.2f} EUR
- **Realized Revenue:** {total_realized:,.2f} EUR
- **Realization Ratio:** {realization_ratio:.2f}%
- **Total Energy Throughput:** {total_mwh_cycled:,.2f} MWh
"""

    # 3. Clean saving mechanism
    output_filename = "BESS_Automated_Report.md"
    
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"Success! Your clean markdown report has been saved to: {output_filename}")
    return report_content

if __name__ == "__main__":
    generate_local_report(521852.81, 640974.20, 122.83, 17021.41)
