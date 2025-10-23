# column_mapper.py
# Maps simplified 14-question survey to model's 85 expected columns

import pandas as pd
import numpy as np

def preprocess_chatbot_answers(answers: dict, train_columns: list) -> pd.DataFrame:
    """
    Convert chatbot answers (14 questions) to model format (85 columns)
    Fill unmapped columns with reasonable defaults
    """
    
    # Initialize all columns with 0
    row_data = {col: 0 for col in train_columns}
    
    
    # Q2: Organization size (1-5)
    if 'organization_size' in answers:
        size_map = {
            "1-10 employees": 1,
            "11-50 employees": 2,
            "51-200 employees": 3,
            "201-500 employees": 4,
            "500+ employees": 5
        }
        value = size_map.get(answers['organization_size'], 3)  # default to medium
        row_data['2_How_big_is_your_organization_'] = value
    
    # Q4: Company operating years (1-5)
    if 'company_operating_years' in answers:
        years_map = {
            "Less than 1 year": 1,
            "1-3 years": 2,
            "3-5 years": 3,
            "5-10 years": 4,
            "More than 10 years": 5
        }
        value = years_map.get(answers['company_operating_years'], 3)
        row_data['4_How_long_has_your_company_been_operating_'] = value
    
    # Q6: Unexpected risk frequency (1-4)
    if 'unexpected_risk_frequency' in answers:
        freq_map = {
            "Rarely (once a year or less)": 1,
            "Occasionally (2-4 times a year)": 2,
            "Frequently (monthly)": 3,
            "Very frequently (weekly)": 4
        }
        value = freq_map.get(answers['unexpected_risk_frequency'], 2)
        row_data['6_How_often_will_your_company_encounter_unexpected_risks_'] = value
    
    # Q8: Project delay frequency (1-5)
    if 'project_delay_frequency' in answers:
        delay_map = {
            "Never": 1,
            "Rarely": 2,
            "Sometimes": 3,
            "Often": 4,
            "Always": 5
        }
        value = delay_map.get(answers['project_delay_frequency'], 3)
        row_data['8_How_often_will_your_company_face_project_delays_due_to_poor_planning_or_changes_in_demand_'] = value
    
    # Q11: Budget overrun response (1-5)
    if 'budget_overrun_response' in answers:
        response_map = {
            "Cut project scope": 1,
            "Request additional budget": 2,
            "Reallocate resources": 3,
            "Extend timeline": 4,
            "Improve team efficiency": 5
        }
        value = response_map.get(answers['budget_overrun_response'], 3)
        row_data['11_How_does_your_team_usually_respond_when_the_project_exceeds_the_budget_or_schedule_'] = value
    
    # Q12: AI tools help (1-5)
    if 'ai_tools_help' in answers:
        ai_map = {
            "Definitely yes": 1,
            "Probably yes": 2,
            "Not sure": 3,
            "Probably no": 4,
            "Definitely no": 5
        }
        value = ai_map.get(answers['ai_tools_help'], 2)
        row_data['12_Do_you_think_using_AI_tools_can_help_your_company_better_manage_cost_and_time_risks_'] = value
    
    # Q14 (Q8 in our survey): Risk severity - scale_multiple
    if 'risk_severity' in answers and isinstance(answers['risk_severity'], dict):
        severity = answers['risk_severity']
        if 'financial_risk_severity' in severity:
            row_data['14_With_1_to_5_as_the_level_how_severe_is_the_following_risks_affecting_your_organization_Financial_risk'] = severity['financial_risk_severity']
        if 'operational_risk_severity' in severity:
            row_data['14_Operational_Risk'] = severity['operational_risk_severity']
        if 'strategic_risk_severity' in severity:
            row_data['14_Strategic_Risk'] = severity['strategic_risk_severity']
        if 'technical_risk_severity' in severity:
            row_data['14_Technical_risks'] = severity['technical_risk_severity']
        if 'compliance_risk_severity' in severity:
            row_data['14_Compliance_risk'] = severity['compliance_risk_severity']
    
    # Q15-18: Individual risk impacts
    if 'financial_risk_impact' in answers:
        row_data['15_With_1_to_5_as_the_level_how_serious_is_financial_risks_such_as_budget_overspending_funding_issues_affecting_your_organization_'] = answers['financial_risk_impact']
    
    if 'operational_risk_impact' in answers:
        row_data['16_How_severe_is_the_impact_of_operational_risks_such_as_inefficient_process_poor_management_on_your_organization_on_the_level_of_1_to_5_'] = answers['operational_risk_impact']
    
    if 'strategic_risk_impact' in answers:
        row_data['17_With_1_to_5_as_the_level_how_severe_is_strategic_risks_such_as_market_competition_project_failure_affecting_your_organization_'] = answers['strategic_risk_impact']
    
    if 'compliance_risk_impact' in answers:
        row_data['18_How_severe_is_compliance_risks_such_as_legal_and_regulatory_issues_affecting_your_organization_on_the_scale_of_1_to_5_'] = answers['compliance_risk_impact']
    
    # Q20: Risk evaluation method (1-5)
    if 'risk_evaluation_method' in answers:
        eval_map = {
            "Formal risk assessment framework": 1,
            "Regular team discussions": 2,
            "Ad-hoc basis when issues arise": 3,
            "Using risk management software": 4,
            "No formal evaluation process": 5
        }
        value = eval_map.get(answers['risk_evaluation_method'], 2)
        row_data['20_How_does_your_organization_evaluate_and_classify_risks_'] = value
    
    # Q22: Strategy effectiveness (1-5)
    if 'strategy_effectiveness' in answers:
        effect_map = {
            "Very effective": 1,
            "Somewhat effective": 2,
            "Neutral": 3,
            "Not very effective": 4,
            "Not effective at all": 5
        }
        value = effect_map.get(answers['strategy_effectiveness'], 2)
        row_data['22_How_effective_do_you_think_your_current_risk_management_strategy_is_'] = value
    
    # Fill remaining unmapped columns with reasonable defaults based on patterns
    # These are checkbox/multiple choice questions we're not asking
    
    # Q3: Software development types - set web development as default
    row_data['3_What_type_of_software_development_does_your_company_focus_on_Multiple_choice_Web_page_development_'] = 1
    
    # Q5: Usual risks - set common ones
    row_data['5_What_types_of_risks_does_your_software_company_usually_face_Multiple_choice_Financial_risk_budget_overspending_funding_issues_'] = 1
    row_data['5_Technical_risk_system_failure_network_security_threat_'] = 1
    
    # Q10: Cost control strategies - set common practices
    row_data['10_What_strategies_does_your_organization_use_to_control_project_costs_and_durations_Multiple_choice_Strict_budget_and_financial_tracking_'] = 1
    
    # Q21: Risk management strategies
    row_data['21_What_risk_management_strategies_does_your_company_currently_adopt_Multiple_choice_Regular_risk_audit_'] = 1
    
    # Q24-27: AI questions - set positive defaults
    row_data['24_Do_you_understand_risk_management_solutions_based_on_artificial_intelligence_'] = 2
    row_data['25_Are_you_willing_to_use_artificial_intelligence_based_tools_for_risk_assessment_and_mitigation_'] = 2
    row_data['27_Are_you_willing_to_participate_in_the_AI_driven_risk_management_pilot_project_'] = 2
    
    # Create DataFrame
    df = pd.DataFrame([row_data], columns=train_columns)
    
    return df