from database import SessionLocal
import models

def clear_database():
    db = SessionLocal()
    try:
        # Delete all rows from your main table
        db.query(models.SurveyResponseDB).delete()
        db.commit()
        print("✅ All data removed from survey_responses table successfully.")
    except Exception as e:
        db.rollback()
        print(f"❌ Error clearing table: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    clear_database()
