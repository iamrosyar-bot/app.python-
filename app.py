"""
app.py
Smart Healthcare Disease Prediction System with AI Chatbot
Flask REST API & Web Application Server
Configured to run strictly from the root directory with no subdirectories.
"""

import sys
sys.dont_write_bytecode = True

import os
import json
import sqlite3
import datetime
from flask import Flask, request, jsonify, send_from_directory, make_response
from flask_cors import CORS

import prediction

DATABASE_FILE = "healthcare.db"

app = Flask(__name__, static_folder=".", template_folder=".")
CORS(app)


# -------------------------------------------------------------
# Database Management
# -------------------------------------------------------------
def get_db_connection():
    """Returns a SQLite connection with row factory for dictionary access."""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes SQLite database tables if they do not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT,
            age INTEGER NOT NULL,
            gender TEXT,
            body_temp REAL NOT NULL,
            systolic_bp REAL,
            diastolic_bp REAL,
            heart_rate REAL,
            blood_glucose REAL,
            symptoms TEXT NOT NULL,
            predicted_category TEXT NOT NULL,
            risk_level TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            top_predictions TEXT,
            explanation TEXT,
            recommendations TEXT,
            vital_warnings TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_message TEXT,
            bot_reply TEXT,
            is_emergency INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()
    print("Database initialized successfully.")


# -------------------------------------------------------------
# Static Web Routing (Direct from root directory)
# -------------------------------------------------------------
@app.route("/")
def serve_index():
    """Serves index.html directly from the root directory."""
    return send_from_directory(".", "index.html")


@app.route("/<path:filename>")
def serve_static_root(filename):
    """Safely serves CSS, JS, and asset files directly from the root directory."""
    safe_extensions = (".html", ".css", ".js", ".json", ".csv", ".ico", ".png", ".svg", ".jpg")
    if filename.endswith(safe_extensions) and os.path.isfile(filename):
        return send_from_directory(".", filename)
    return jsonify({"error": "Resource not found"}), 404


# -------------------------------------------------------------
# Chatbot AI & Healthcare Knowledge Base Engine
# -------------------------------------------------------------
EMERGENCY_KEYWORDS = [
    "chest pain", "crushing chest", "heart attack", "difficulty breathing",
    "cannot breathe", "can't breathe", "shortness of breath with chest",
    "stroke", "face droop", "slurred speech", "unconscious", "unconsciousness",
    "passed out", "fainted suddenly", "severe bleeding", "coughing blood",
    "anaphylaxis", "throat closing", "swollen tongue", "severe allergic",
    "suicide", "overdose", "poison", "cyanosis", "blue lips"
]

FAQ_KNOWLEDGE_BASE = [
    {
        "keywords": ["what is this", "about", "how does it work", "who are you", "what can you do"],
        "category": "About",
        "reply": "I am the **HealthCare AI Assistant**, your personal health risk educator. I can provide evidence-based educational insights regarding symptoms, lifestyle habits, preventive health metrics, and help explain your disease risk assessment results. Please note that I am an educational guide, not a licensed doctor.",
        "suggestions": ["How do I check my symptoms?", "What causes fever?", "Healthy diet tips", "Emergency signs"]
    },
    {
        "keywords": ["fever", "high temperature", "temperature", "chills"],
        "category": "Symptoms",
        "reply": "A **fever** is defined clinically as a core body temperature of 100.4°F (38°C) or higher. It is a natural physiological defense mechanism indicating that your immune system is combating an infection (often viral or bacterial).\n\n**Home Care:**\n- Stay well hydrated with water, broths, and electrolyte drinks.\n- Rest in a comfortable, temperate environment.\n- Wear lightweight clothing.\n\n**When to see a doctor:** If fever exceeds 103°F (39.4°C), lasts more than 3 consecutive days, or is accompanied by stiff neck, shortness of breath, confusion, or rash.",
        "suggestions": ["Why do people experience headaches?", "What causes fatigue?", "Dehydration symptoms", "Start Risk Assessment"]
    },
    {
        "keywords": ["diabetes", "blood sugar", "glucose", "hyperglycemia", "high sugar"],
        "category": "Chronic Care",
        "reply": "**Common Diabetes Warning Signs:**\n1. **Frequent Urination (Polyuria):** Waking up multiple times at night to urinate.\n2. **Excessive Thirst (Polydipsia):** Constant thirst even after drinking plenty of water.\n3. **Unexplained Weight Loss:** Losing weight despite normal or increased eating.\n4. **Chronic Fatigue:** Feeling drained because cells lack accessible glucose.\n5. **Blurred Vision & Slow-Healing Cuts.**\n\n**Clinical Thresholds:** Normal fasting glucose is typically 70-99 mg/dL. Fasting glucose of 100-125 mg/dL indicates prediabetes, while 126 mg/dL or higher on repeat tests indicates diabetes. Consult your physician for an HbA1c test.",
        "suggestions": ["Blood pressure concerns", "Healthy diet habits", "Explain metabolic risk", "Exercise recommendations"]
    },
    {
        "keywords": ["blood pressure", "hypertension", "high bp", "systolic", "diastolic"],
        "category": "Cardiovascular",
        "reply": "**Blood Pressure Reference Categories:**\n- **Normal:** Systolic < 120 and Diastolic < 80 mmHg\n- **Elevated:** Systolic 120-129 and Diastolic < 80 mmHg\n- **Stage 1 Hypertension:** Systolic 130-139 or Diastolic 80-89 mmHg\n- **Stage 2 Hypertension:** Systolic 140+ or Diastolic 90+ mmHg\n- **Hypertensive Crisis:** Systolic > 180 and/or Diastolic > 120 mmHg (Requires immediate emergency care!)\n\n**Management:** Reduce dietary sodium (<2,300 mg/day), engage in daily aerobic walking, manage stress, and follow medication prescribed by your physician.",
        "suggestions": ["What causes chest discomfort?", "Diet for high blood pressure", "Dizziness causes", "Check my health risk"]
    },
    {
        "keywords": ["headache", "migraine", "head hurt", "head throbbing"],
        "category": "Neurological",
        "reply": "**Common Causes of Headaches:**\n- **Tension Headaches:** Stress, poor neck posture, screen glare, eye fatigue.\n- **Dehydration & Caffeine:** Missing water intake or caffeine withdrawal.\n- **Migraines:** Neurovascular episodes often with throbbing unilateral pain, nausea, and light/sound sensitivity.\n- **Sinus Pressure:** Congestion from colds or allergies.\n\n**Immediate Red Flags:** Sudden 'thunderclap' headache reaching peak intensity in seconds, headache with fever and stiff neck, or headache accompanied by numbness or speech difficulties require urgent emergency evaluation.",
        "suggestions": ["Dehydration symptoms", "How to manage migraines", "What causes fatigue?", "Sleep hygiene tips"]
    },
    {
        "keywords": ["fatigue", "tired", "exhaustion", "low energy", "weakness"],
        "category": "General Health",
        "reply": "**Common Causes of Fatigue:**\n- Inadequate sleep quality or inconsistent sleep schedules.\n- Nutritional deficiencies (iron deficiency anemia, vitamin D or B12 deficiency).\n- Dehydration and sedentary lifestyle.\n- Chronic emotional or mental stress.\n- Endocrine imbalances such as hypothyroidism or elevated blood glucose.\n\n**Next Steps:** Aim for 7-9 hours of consistent sleep, drink 2-3 liters of water, and consider routine blood tests (CBC, Ferritin, TSH, HbA1c) if fatigue persists for more than two weeks.",
        "suggestions": ["Sleep hygiene tips", "Healthy diet habits", "Check diabetes warning signs", "Hydration tips"]
    },
    {
        "keywords": ["dehydration", "thirsty", "dry mouth", "water intake", "hydration"],
        "category": "Wellness",
        "reply": "**Warning Signs of Dehydration:**\n- Dark amber-colored urine (healthy urine is pale straw yellow).\n- Dry sticky mouth, parched lips, and reduced urination frequency.\n- Lightheadedness, dizziness upon standing, and headache.\n- Decreased skin elasticity and general sluggishness.\n\n**Recommended Intake:** The baseline recommendation is roughly 2 to 3 liters (8-10 glasses) of fluid daily for adults, adjusting upwards in hot weather or during rigorous exercise.",
        "suggestions": ["Healthy lifestyle habits", "Why do I feel dizzy?", "Exercise recommendations", "Start Risk Assessment"]
    },
    {
        "keywords": ["diet", "food", "nutrition", "healthy eating", "what to eat"],
        "category": "Nutrition",
        "reply": "**Evidence-Based Healthy Nutrition Guidelines:**\n1. **Focus on Whole Foods:** Build meals around vegetables, legumes, whole grains, nuts, and clean proteins.\n2. **Healthy Fats:** Incorporate extra virgin olive oil, avocados, seeds, and omega-3 rich fish.\n3. **Limit Added Sugars:** Minimize sodas, commercial juices, pastries, and sweetened snacks.\n4. **Portion Sodium:** Keep sodium under 2,000-2,300 mg per day for cardiovascular protection.\n5. **Fiber First:** Target 25-35 grams of dietary fiber daily for optimal gut microbiome and metabolic control.",
        "suggestions": ["Exercise recommendations", "Diabetes warning signs", "Sleep awareness", "Hydration tips"]
    },
    {
        "keywords": ["exercise", "workout", "physical activity", "fitness", "cardio"],
        "category": "Fitness",
        "reply": "**Recommended Physical Activity Standards:**\n- **Aerobic Activity:** At least 150 minutes of moderate aerobic exercise (brisk walking, cycling, swimming) or 75 minutes of vigorous exercise weekly.\n- **Strength Training:** Muscle-strengthening activities involving major muscle groups at least 2 days per week.\n- **Daily Movement:** Break up prolonged sitting every 45-60 minutes with a 2-minute stretch or walk.\n\n*Safety Note:* If you have heart disease, severe joint pain, or haven't exercised recently, consult your physician before starting an intensive routine.",
        "suggestions": ["Healthy diet habits", "Blood pressure concerns", "Sleep awareness", "Check my health risk"]
    },
    {
        "keywords": ["sleep", "insomnia", "sleeping", "rest", "tired in morning"],
        "category": "Sleep",
        "reply": "**Science-Backed Sleep Hygiene Practices:**\n1. **Fixed Schedule:** Sleep and wake at the same times every day, even on weekends.\n2. **Cool & Dark Environment:** Keep your bedroom quiet, dark, and between 65-68°F (18-20°C).\n3. **Digital Sunset:** Discontinue screens (phones, tablets, TV) 60 minutes before bedtime.\n4. **Caffeine Cutoff:** Avoid caffeine at least 6-8 hours prior to sleep.\n5. **Wind-Down Ritual:** Practice 10 minutes of gentle stretching, reading, or deep diaphragmatic breathing.",
        "suggestions": ["What causes fatigue?", "Dehydration symptoms", "Healthy diet habits", "Start Risk Assessment"]
    },
    {
        "keywords": ["cough", "bronchitis", "phlegm", "dry cough", "chest cough"],
        "category": "Respiratory",
        "reply": "**Cough Overview:**\n- **Acute Cough (<3 weeks):** Commonly triggered by viral upper respiratory infections (common cold, flu), post-nasal drip, or acute bronchitis.\n- **Chronic Cough (>8 weeks):** Often linked to asthma, gastroesophageal reflux disease (GERD), allergies, or environmental pollutants.\n\n**Soothing Tips:** Warm herbal teas with honey (for adults and children >1 year), warm steam inhalation, and elevating your head while sleeping.\n\n**Seek Prompt Care If:** Coughing up blood, wheezing, experiencing chest pain, or having difficulty taking a deep breath.",
        "suggestions": ["What causes fever?", "What is asthma alert?", "Explain my prediction result", "Emergency signs"]
    },
    {
        "keywords": ["explain prediction", "result", "risk category", "what does my assessment mean"],
        "category": "Prediction Guidance",
        "reply": "Your **Disease Risk Assessment** analyzes your entered vital signs and symptom combinations using our Scikit-Learn machine learning model.\n\n- **Low Risk:** Vitals and symptoms match typical mild or self-limiting conditions. Routine self-care and monitoring are advised.\n- **Moderate Risk:** Symptoms require close observation and an elective consultation with a healthcare professional.\n- **High Risk:** Elevated vital signs or notable symptom clusters suggest potential pathology needing timely medical evaluation.\n- **Urgent:** Critical vital thresholds or red-flag symptom combinations (such as chest tightness + breathlessness) warrant immediate emergency medical intervention.",
        "suggestions": ["Start Disease Prediction", "View Health History", "Blood pressure concerns", "Diabetes warning signs"]
    }
]


def process_chat_message(message):
    """
    Evaluates incoming message for emergencies, matches curated healthcare knowledge base,
    and returns safe, educational response with interactive suggestion pills.
    """
    clean_msg = message.strip().lower()

    # 1. Critical Emergency Detection
    for kw in EMERGENCY_KEYWORDS:
        if kw in clean_msg:
            return {
                "reply": "🚨 **CRITICAL MEDICAL EMERGENCY ALERT**\n\nThe symptoms or concerns you described may indicate a life-threatening medical emergency (such as cardiac distress, stroke, respiratory failure, or severe anaphylaxis).\n\n**IMMEDIATE ACTIONS REQUIRED:**\n1. **Call emergency services immediately** (Dial 911 in the US/Canada, 112 in Europe, 999 in the UK, or your local emergency number).\n2. **Proceed to the nearest Emergency Room (ER) or hospital.**\n3. Do NOT wait for online replies or attempt home remedies.\n4. If someone is with you, notify them immediately so they can assist you.",
                "is_emergency": True,
                "category": "Emergency",
                "suggestions": ["Call 911 / Local Emergency", "Find Nearest Hospital", "Ask another non-emergency question"]
            }

    # 2. Check Curated Knowledge Base
    best_match = None
    best_score = 0

    for item in FAQ_KNOWLEDGE_BASE:
        score = sum(1 for kw in item["keywords"] if kw in clean_msg)
        if score > best_score:
            best_score = score
            best_match = item

    if best_match and best_score > 0:
        return {
            "reply": best_match["reply"],
            "is_emergency": False,
            "category": best_match["category"],
            "suggestions": best_match["suggestions"]
        }

    # 3. Optional External AI Fallback (OpenAI or Gemini API if environment variable set)
    # Check if OPENAI_API_KEY is provided
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        try:
            import urllib.request
            req_data = json.dumps({
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "You are HealthCare AI Assistant, an educational medical assistant. Provide clear, empathetic, evidence-based healthcare education. Always include a safety reminder that you are not a doctor and cannot diagnose, prescribe medication, or provide clinical dosages. If an emergency is suspected, advise calling 911."},
                    {"role": "user", "content": message}
                ],
                "max_tokens": 300,
                "temperature": 0.5
            }).encode('utf-8')
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=req_data,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {openai_key}"}
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                reply_text = data["choices"][0]["message"]["content"]
                return {
                    "reply": reply_text + "\n\n*Note: This information is for educational purposes and is not a substitute for professional medical diagnosis.*",
                    "is_emergency": False,
                    "category": "AI Assistant",
                    "suggestions": ["What causes fever?", "Diabetes warning signs", "Start Risk Assessment"]
                }
        except Exception:
            pass  # Fallback to standard safe response

    # 4. Safe Default Guidance
    return {
        "reply": f"Thank you for reaching out. While I don't have a specific medical article matching *'{message}'*, I can help you with:\n\n- Exploring causes of common symptoms (fever, headache, fatigue, cough, nausea)\n- Understanding chronic disease risk indicators (diabetes, hypertension)\n- Healthy lifestyle, nutrition, hydration, and sleep guidelines\n- Explaining your disease prediction assessment results\n\nFor personal medical concerns or persistent symptoms, please consult a qualified healthcare professional.",
        "is_emergency": False,
        "category": "General Advice",
        "suggestions": [
            "What causes fever?",
            "Blood pressure concerns",
            "Diabetes warning signs",
            "Healthy diet habits",
            "Start Risk Assessment"
        ]
    }


# -------------------------------------------------------------
# REST API Endpoints
# -------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def api_health():
    """
    Returns system status, database status, model evaluation metrics,
    and aggregate stats.
    """
    try:
        model_metrics = prediction.get_model_status()
        conn = get_db_connection()
        total_assessments = conn.execute("SELECT COUNT(*) as count FROM assessments").fetchone()["count"]
        conn.close()

        return jsonify({
            "status": "healthy",
            "system": "Smart Healthcare Disease Prediction System",
            "version": "2.0.0",
            "database": "connected",
            "total_assessments": total_assessments,
            "model_metrics": model_metrics,
            "timestamp": datetime.datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Predicts disease risk category based on patient vitals and selected symptoms.
    Saves record to SQLite database and returns structured risk report.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON payload"}), 400

    # Input validations
    age = data.get("age")
    body_temp = data.get("body_temp")

    if age is None:
        return jsonify({"error": "Patient Age is required."}), 400
    try:
        age_val = int(age)
        if age_val < 1 or age_val > 120:
            return jsonify({"error": "Age must be between 1 and 120."}), 400
    except ValueError:
        return jsonify({"error": "Age must be a valid integer."}), 400

    if body_temp is None:
        return jsonify({"error": "Body temperature is required."}), 400
    try:
        temp_val = float(body_temp)
        if temp_val < 30.0 or temp_val > 115.0:
            return jsonify({"error": "Body temperature must be in a realistic physiological range (e.g. 96 - 105°F or 35 - 41°C)."}), 400
    except ValueError:
        return jsonify({"error": "Body temperature must be a valid number."}), 400

    patient_name = str(data.get("patient_name", "Anonymous")).strip() or "Anonymous"
    gender = str(data.get("gender", "Other")).strip() or "Other"

    try:
        # Run ML inference & clinical analysis
        result = prediction.predict_disease(data)

        # Save to SQLite
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO assessments (
                patient_name, age, gender, body_temp, systolic_bp, diastolic_bp,
                heart_rate, blood_glucose, symptoms, predicted_category, risk_level,
                confidence_score, top_predictions, explanation, recommendations, vital_warnings
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            patient_name,
            age_val,
            gender,
            float(result["vitals_summary"]["body_temp"].split()[0]),
            float(result["vitals_summary"]["blood_pressure"].split("/")[0]),
            float(result["vitals_summary"]["blood_pressure"].split("/")[1].split()[0]),
            float(result["vitals_summary"]["heart_rate"].split()[0]),
            float(result["vitals_summary"]["blood_glucose"].split()[0]),
            json.dumps(result["selected_symptoms"]),
            result["predicted_category"],
            result["risk_level"],
            result["confidence_score"],
            json.dumps(result["top_predictions"]),
            result["explanation"],
            json.dumps(result["recommendations"]),
            json.dumps(result["vital_warnings"])
        ))

        record_id = cursor.lastrowid
        conn.commit()
        conn.close()

        result["id"] = record_id
        result["patient_name"] = patient_name
        result["created_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return jsonify({
            "success": True,
            "message": "Assessment processed and saved successfully.",
            "data": result
        }), 201

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Handles interactive chatbot inquiries, performs emergency keyword screening,
    and returns AI or FAQ response.
    """
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field"}), 400

    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify({"error": "Message cannot be empty"}), 400

    session_id = str(data.get("session_id", "default_session"))

    response_data = process_chat_message(message)

    # Log chat safely
    try:
        conn = get_db_connection()
        conn.execute("""
            INSERT INTO chat_logs (session_id, user_message, bot_reply, is_emergency)
            VALUES (?, ?, ?, ?)
        """, (session_id, message, response_data["reply"], 1 if response_data["is_emergency"] else 0))
        conn.commit()
        conn.close()
    except Exception:
        pass  # Logging error should not break chat response

    return jsonify(response_data), 200


@app.route("/api/history", methods=["GET"])
def api_get_history():
    """
    Retrieves patient health assessment history with search, filter, and summary stats.
    """
    search = request.args.get("search", "").strip()
    risk_filter = request.args.get("risk_level", "").strip()

    conn = get_db_connection()
    query = "SELECT * FROM assessments WHERE 1=1"
    params = []

    if search:
        query += " AND (patient_name LIKE ? OR predicted_category LIKE ? OR symptoms LIKE ?)"
        search_param = f"%{search}%"
        params.extend([search_param, search_param, search_param])

    if risk_filter and risk_filter.lower() != "all":
        query += " AND risk_level = ?"
        params.append(risk_filter)

    query += " ORDER BY created_at DESC"

    rows = conn.execute(query, params).fetchall()

    records = []
    risk_counts = {"Low": 0, "Moderate": 0, "High": 0, "Urgent": 0}
    symptom_frequency = {}

    for row in rows:
        sym_list = json.loads(row["symptoms"]) if row["symptoms"] else []
        for s in sym_list:
            symptom_frequency[s] = symptom_frequency.get(s, 0) + 1

        r_level = row["risk_level"]
        if r_level in risk_counts:
            risk_counts[r_level] += 1

        records.append({
            "id": row["id"],
            "patient_name": row["patient_name"],
            "age": row["age"],
            "gender": row["gender"],
            "body_temp": row["body_temp"],
            "blood_pressure": f"{int(row['systolic_bp'])}/{int(row['diastolic_bp'])}" if row["systolic_bp"] else "N/A",
            "heart_rate": row["heart_rate"],
            "blood_glucose": row["blood_glucose"],
            "symptoms": sym_list,
            "predicted_category": row["predicted_category"],
            "risk_level": row["risk_level"],
            "confidence_score": row["confidence_score"],
            "top_predictions": json.loads(row["top_predictions"]) if row["top_predictions"] else [],
            "explanation": row["explanation"],
            "recommendations": json.loads(row["recommendations"]) if row["recommendations"] else [],
            "vital_warnings": json.loads(row["vital_warnings"]) if row["vital_warnings"] else [],
            "created_at": row["created_at"]
        })

    conn.close()

    return jsonify({
        "total": len(records),
        "risk_counts": risk_counts,
        "symptom_frequency": symptom_frequency,
        "records": records
    }), 200


@app.route("/api/history/<int:record_id>", methods=["DELETE"])
def api_delete_history_item(record_id):
    """Deletes a single assessment record by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM assessments WHERE id = ?", (record_id,))
    deleted = cursor.rowcount
    conn.commit()
    conn.close()

    if deleted > 0:
        return jsonify({"success": True, "message": f"Record #{record_id} deleted successfully."}), 200
    return jsonify({"error": "Record not found"}), 404


@app.route("/api/history", methods=["DELETE"])
def api_clear_history():
    """Clears all health assessment history records."""
    conn = get_db_connection()
    conn.execute("DELETE FROM assessments")
    conn.commit()
    conn.close()
    return jsonify({"success": True, "message": "All assessment history cleared successfully."}), 200


@app.route("/api/report/<int:record_id>", methods=["GET"])
def api_download_report(record_id):
    """
    Generates a structured medical assessment summary report.
    Can be downloaded directly as formatted text or inspected as JSON.
    """
    format_type = request.args.get("format", "text")
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM assessments WHERE id = ?", (record_id,)).fetchone()
    conn.close()

    if not row:
        return jsonify({"error": "Assessment record not found"}), 404

    symptoms = json.loads(row["symptoms"]) if row["symptoms"] else []
    recommendations = json.loads(row["recommendations"]) if row["recommendations"] else []
    vital_warnings = json.loads(row["vital_warnings"]) if row["vital_warnings"] else []

    if format_type == "json":
        return jsonify(dict(row)), 200

    # Format professional clinical text report
    report_lines = [
        "=" * 60,
        "  SMART HEALTHCARE DISEASE RISK PREDICTION REPORT",
        "  HealthCare AI System - Educational Health Assessment",
        "=" * 60,
        f"Report ID: #{row['id']}",
        f"Date/Time: {row['created_at']}",
        f"Patient:   {row['patient_name']} (Age: {row['age']}, Gender: {row['gender']})",
        "-" * 60,
        "VITAL SIGNS SUMMARY:",
        f"  • Body Temperature: {row['body_temp']} °F",
        f"  • Blood Pressure:   {row['systolic_bp']}/{row['diastolic_bp']} mmHg",
        f"  • Heart Rate:       {row['heart_rate']} bpm",
        f"  • Blood Glucose:    {row['blood_glucose']} mg/dL",
        "-" * 60,
        "REPORTED SYMPTOMS:",
        f"  {', '.join(symptoms) if symptoms else 'No active symptoms reported.'}",
        "-" * 60,
        "RISK ASSESSMENT OUTCOME:",
        f"  • Risk Category:    {row['predicted_category']}",
        f"  • Assigned Risk:    {row['risk_level'].upper()}",
        f"  • Model Confidence: {row['confidence_score']:.1f}%",
        "",
        "CLINICAL EXPLANATION:",
        f"  {row['explanation']}",
        "-" * 60,
        "VITAL SIGNS ALERTS:",
    ]

    if vital_warnings:
        for w in vital_warnings:
            report_lines.append(f"  [!] {w}")
    else:
        report_lines.append("  All recorded vitals fall within expected baseline thresholds.")

    report_lines.extend([
        "-" * 60,
        "RECOMMENDED NEXT STEPS & GUIDANCE:",
    ])
    for idx, rec in enumerate(recommendations, 1):
        report_lines.append(f"  {idx}. {rec}")

    report_lines.extend([
        "=" * 60,
        "CRITICAL MEDICAL DISCLAIMER:",
        "This tool provides educational health-risk information and is NOT",
        "a substitute for professional medical diagnosis, advice, or treatment.",
        "Always consult a licensed physician or emergency medical services.",
        "=" * 60
    ])

    report_content = "\n".join(report_lines)
    response = make_response(report_content)
    response.headers["Content-Type"] = "text/plain; charset=utf-8"
    response.headers["Content-Disposition"] = f"attachment; filename=Health_Assessment_Report_{row['id']}.txt"
    return response


# -------------------------------------------------------------
# Server Entry Point
# -------------------------------------------------------------
if __name__ == "__main__":
    init_db()
    # Ensure model is initialized
    prediction.get_model_status()
    print("Starting Smart Healthcare Application on http://127.0.0.1:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=False)
