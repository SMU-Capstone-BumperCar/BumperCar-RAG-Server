from flask import Blueprint, jsonify, request
from app.service.hospital_review_service import summarize_hospital_review 

hospital_review_bp = Blueprint('hospital_review', __name__)

@hospital_review_bp.route('/api/summarize-review', methods=['POST'])
def summarize_review():
    data = request.get_json()
    query = data.get("query", "")
    
    if not query:
        return jsonify({"error": "질문이 필요합니다."}), 400

    try:
        answer = summarize_hospital_review(query)
        return jsonify({"answer": answer}), 200
    except Exception as e:
        return jsonify({"error": f"오류가 발생했습니다: {str(e)}"}), 500
