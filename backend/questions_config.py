QUESTIONS = [
    {
        "id": "q1",
        "key": "company_name",
        "question": "What is your software company name?",
        "type": "text",
        "required": False,
        "metadata": True
    },
    {
        "id": "q2",
        "key": "organization_size",
        "question": "How big is your organization?",
        "type": "radio",
        "options": [
            "1-10 employees",
            "11-50 employees",
            "51-200 employees",
            "201-500 employees",
            "500+ employees"
        ],
        "required": True
    },
    {
        "id": "q3",
        "key": "company_operating_years",
        "question": "How long has your company been operating?",
        "type": "radio",
        "options": [
            "Less than 1 year",
            "1-3 years",
            "3-5 years",
            "5-10 years",
            "More than 10 years"
        ],
        "required": True
    },
    {
        "id": "q4",
        "key": "unexpected_risk_frequency",
        "question": "How often does your company encounter unexpected risks?",
        "type": "radio",
        "options": [
            "Rarely (once a year or less)",
            "Occasionally (2-4 times a year)",
            "Frequently (monthly)",
            "Very frequently (weekly)"
        ],
        "required": True
    },
    {
        "id": "q5",
        "key": "project_delay_frequency",
        "question": "How often does your company face project delays due to poor planning or changes in demand?",
        "type": "radio",
        "options": [
            "Never",
            "Rarely",
            "Sometimes",
            "Often",
            "Always"
        ],
        "required": True
    },
    {
        "id": "q6",
        "key": "budget_overrun_response",
        "question": "How does your team usually respond when the project exceeds the budget or schedule?",
        "type": "radio",
        "options": [
            "Cut project scope",
            "Request additional budget",
            "Reallocate resources",
            "Extend timeline",
            "Improve team efficiency"
        ],
        "required": True
    },
    {
        "id": "q7",
        "key": "ai_tools_help",
        "question": "Do you think using AI tools can help your company better manage cost and time risks?",
        "type": "radio",
        "options": [
            "Definitely yes",
            "Probably yes",
            "Not sure",
            "Probably no",
            "Definitely no"
        ],
        "required": True
    },
    {
        "id": "q8",
        "key": "risk_severity",
        "question": "On a scale of 1 to 5, how severe are the following risks affecting your organization?",
        "type": "scale_multiple",
        "sub_questions": [
            {"key": "financial_risk_severity", "label": "Financial risk"},
            {"key": "operational_risk_severity", "label": "Operational risk"},
            {"key": "strategic_risk_severity", "label": "Strategic risk"},
            {"key": "technical_risk_severity", "label": "Technical risks"},
            {"key": "compliance_risk_severity", "label": "Compliance risk"}
        ],
        "scale": {"min": 1, "max": 5},
        "required": True
    },
    {
        "id": "q9",
        "key": "financial_risk_impact",
        "question": "On a scale of 1 to 5, how serious are financial risks (budget overspending, funding issues) affecting your organization?",
        "type": "scale",
        "scale": {"min": 1, "max": 5},
        "required": True
    },
    {
        "id": "q10",
        "key": "operational_risk_impact",
        "question": "On a scale of 1 to 5, how severe is the impact of operational risks (inefficient process, poor management)?",
        "type": "scale",
        "scale": {"min": 1, "max": 5},
        "required": True
    },
    {
        "id": "q11",
        "key": "strategic_risk_impact",
        "question": "On a scale of 1 to 5, how severe are strategic risks (market competition, project failure) affecting your organization?",
        "type": "scale",
        "scale": {"min": 1, "max": 5},
        "required": True
    },
    {
        "id": "q12",
        "key": "compliance_risk_impact",
        "question": "On a scale of 1 to 5, how severe are compliance risks (legal and regulatory issues)?",
        "type": "scale",
        "scale": {"min": 1, "max": 5},
        "required": True
    },
    {
        "id": "q13",
        "key": "risk_evaluation_method",
        "question": "How does your organization evaluate and classify risks?",
        "type": "radio",
        "options": [
            "Formal risk assessment framework",
            "Regular team discussions",
            "Ad-hoc basis when issues arise",
            "Using risk management software",
            "No formal evaluation process"
        ],
        "required": True
    },
    {
        "id": "q14",
        "key": "strategy_effectiveness",
        "question": "How effective do you think your current risk management strategy is?",
        "type": "radio",
        "options": [
            "Very effective",
            "Somewhat effective",
            "Neutral",
            "Not very effective",
            "Not effective at all"
        ],
        "required": True
    }
]

def get_question_by_id(question_id):
    for q in QUESTIONS:
        if q["id"] == question_id:
            return q
    return None

def get_survey_questions():
    return [q for q in QUESTIONS if not q.get("metadata", False)]

def get_metadata_questions():
    return [q for q in QUESTIONS if q.get("metadata", False)]









# questions_config_nlp.py
# NLP-focused questions that encourage free-text responses

# NLP_QUESTIONS = [
#     {
#         "id": "q1",
#         "key": "company_name",
#         "question": "What is your company name?",
#         "type": "text",
#         "required": False,
#         "metadata": True
#     },
#     {
#         "id": "q2",
#         "key": "company_description",
#         "question": "Tell me about your company. What do you do and how big is your team?",
#         "type": "text_long",
#         "placeholder": "e.g., We're a 50-person software company building mobile apps...",
#         "required": True,
#         "nlp_analyze": True
#     },
#     {
#         "id": "q3",
#         "key": "main_challenges",
#         "question": "What are the main challenges your company is currently facing?",
#         "type": "text_long",
#         "placeholder": "Describe your biggest concerns, struggles, or obstacles...",
#         "required": True,
#         "nlp_analyze": True
#     },
#     {
#         "id": "q4",
#         "key": "recent_incidents",
#         "question": "Have you experienced any recent incidents or setbacks? If so, please describe.",
#         "type": "text_long",
#         "placeholder": "e.g., We had a server outage last month that cost us customers...",
#         "required": False,
#         "nlp_analyze": True
#     },
#     {
#         "id": "q5",
#         "key": "financial_situation",
#         "question": "How would you describe your company's financial health and budget management?",
#         "type": "text_long",
#         "placeholder": "Talk about cash flow, budget control, funding, expenses...",
#         "required": True,
#         "nlp_analyze": True
#     },
#     {
#         "id": "q6",
#         "key": "team_operations",
#         "question": "Describe how your team works. Are there any operational issues or inefficiencies?",
#         "type": "text_long",
#         "placeholder": "Project delays, communication issues, process problems...",
#         "required": True,
#         "nlp_analyze": True
#     },
#     {
#         "id": "q7",
#         "key": "technology_concerns",
#         "question": "What are your main technology and security concerns?",
#         "type": "text_long",
#         "placeholder": "Infrastructure, cybersecurity, technical debt, system failures...",
#         "required": True,
#         "nlp_analyze": True
#     },
#     {
#         "id": "q8",
#         "key": "market_competition",
#         "question": "How do you view your competitive position in the market?",
#         "type": "text_long",
#         "placeholder": "Market share, competitors, strategic challenges...",
#         "required": True,
#         "nlp_analyze": True
#     },
#     {
#         "id": "q9",
#         "key": "compliance_regulatory",
#         "question": "Do you have any compliance, legal, or regulatory concerns?",
#         "type": "text_long",
#         "placeholder": "Legal requirements, data protection, regulatory issues...",
#         "required": False,
#         "nlp_analyze": True
#     },
#     {
#         "id": "q10",
#         "key": "biggest_worry",
#         "question": "If you had to name ONE thing that keeps you up at night about your business, what would it be?",
#         "type": "text_long",
#         "placeholder": "Be specific about your biggest fear or concern...",
#         "required": True,
#         "nlp_analyze": True
#     },
#     # Optional: Keep some structured questions for hybrid approach
#     {
#         "id": "q11",
#         "key": "risk_severity",
#         "question": "On a scale of 1 to 5, rate these risk areas:",
#         "type": "scale_multiple",
#         "sub_questions": [
#             {"key": "financial_risk_severity", "label": "Financial risk"},
#             {"key": "operational_risk_severity", "label": "Operational risk"},
#             {"key": "strategic_risk_severity", "label": "Strategic risk"},
#             {"key": "technical_risk_severity", "label": "Technical risk"},
#             {"key": "compliance_risk_severity", "label": "Compliance risk"}
#         ],
#         "scale": {"min": 1, "max": 5},
#         "required": True,
#         "nlp_analyze": False
#     }
# ]

# # Separate text and structured responses
# def separate_responses(answers: dict):
#     """Separate text responses (for NLP) from structured data"""
#     text_responses = {}
#     structured_data = {}
    
#     for key, value in answers.items():
#         question = next((q for q in NLP_QUESTIONS if q['key'] == key), None)
        
#         if question and question.get('nlp_analyze'):
#             text_responses[key] = value
#         else:
#             structured_data[key] = value
    
#     return text_responses, structured_data