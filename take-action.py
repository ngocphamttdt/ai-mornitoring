
def generate_llm_prompt(instance_id, metric, current_value, status, priority_score):
    """
    Generates a professional English prompt for the LLM to 'Take Action'
    based on the risk and opportunity analysis.
    """
    
    # Prompt template designed for System Reliability Engineering (SRE)
    prompt = f"""
    Role: Senior System Reliability Engineer (SRE).
    Task: Provide a concise 'Take Action' plan for a server monitoring alert.
    
    System Context:
    - Instance ID: {instance_id}
    - Monitoring Metric: {metric} (e.g., CPU, RAM)
    - Current/Predicted Usage: {current_value}%
    - Classification: {status}
    - Priority Level: {priority_score} (1 is highest, 7 is lowest)
    
    Rules for Action:
    1. If status is 'Critical' or 'Major', focus on immediate mitigation to prevent system failure.
    2. If status is 'High/Medium Opportunity', focus on cost optimization (downsizing).
    3. If status is 'Minor', suggest routine observation.

    Response Requirement: Provide 2-3 specific technical steps in a professional tone.
    """
    return prompt

# Example Usage: A Critical Risk Case
critical_alert = generate_llm_prompt(
    instance_id="Prod-Server-01", 
    metric="CPU Core Usage", 
    current_value=94.5, 
    status="CRITICAL (Offset < 30m)", 
    priority_score=1
)

print(critical_alert)