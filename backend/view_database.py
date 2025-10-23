"""
Automatically export all records from the risk_chatbot database into a CSV file
Run this in the backend folder: python view_database.py
"""

import sqlite3
import json
from datetime import datetime
import pandas as pd
import os

# Connect to database
db_path = 'risk_chatbot.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()


print(" RISK CHATBOT DATABASE VIEWER & AUTO EXPORTER")


# Get all survey responses
cursor.execute("SELECT * FROM survey_responses ORDER BY timestamp DESC")
rows = cursor.fetchall()

if not rows:
    print("\n No records found in database!")
    print("Complete a survey first, then run this script again.")
else:
    print(f"\n Found {len(rows)} record(s)\n")
   
    
    for i, row in enumerate(rows, 1):
        record_id, company_name, answers_json, predicted_risk, confidence, timestamp = row
        
        # print(f"\n📋 RECORD #{i}")
        # print("-" * 80)
        # print(f"ID:               {record_id}")
        # print(f"Company:          {company_name}")
        # print(f"Predicted Risk:   {predicted_risk}")
        # print(f"Confidence:       {confidence:.2%}")
        # print(f"Timestamp:        {timestamp}")
        
        # Parse and display answers
        try:
            answers = json.loads(answers_json)
            print(f"\n Answers ({len(answers)} questions):")
            for key, value in answers.items():
                if isinstance(value, dict):
                    print(f"  • {key}:")
                    for sub_key, sub_val in value.items():
                        print(f"      - {sub_key}: {sub_val}")
                elif isinstance(value, list):
                    print(f"  • {key}: {', '.join(value)}")
                else:
                    print(f"  • {key}: {value}")
        except Exception as e:
            print(f"  Raw answers: {answers_json[:100]}... ({e})")
    


# Count by risk type
cursor.execute("""
    SELECT predicted_risk, COUNT(*) as count, AVG(confidence) as avg_confidence
    FROM survey_responses 
    GROUP BY predicted_risk 
    ORDER BY count DESC
""")
stats = cursor.fetchall()

print("\nRisk Distribution:")
for risk, count, avg_conf in stats:
    print(f"  • {risk}: {count} assessments (avg confidence: {avg_conf:.2%})")

# Companies
cursor.execute("SELECT COUNT(DISTINCT company_name) FROM survey_responses")
unique_companies = cursor.fetchone()[0]
print(f"\nTotal Unique Companies: {unique_companies}")

# Average confidence
cursor.execute("SELECT AVG(confidence) FROM survey_responses")
avg_confidence = cursor.fetchone()[0]
if avg_confidence:
    print(f"Overall Average Confidence: {avg_confidence:.2%}")




export_dir = "exports"
os.makedirs(export_dir, exist_ok=True)
filename = f"{export_dir}/survey_responses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

df = pd.read_sql_query("SELECT * FROM survey_responses", conn)
df.to_csv(filename, index=False)
print(f"\n Auto-export complete! CSV saved to: {filename}")

conn.close()
print("\n Done!")
