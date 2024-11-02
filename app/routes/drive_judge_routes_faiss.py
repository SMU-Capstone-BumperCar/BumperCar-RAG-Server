from flask import Blueprint, jsonify, request
from app.service.drive_judge_service_faiss import analyze_drive_judge  

drive_judge_bp = Blueprint('drive_judge', __name__)

@drive_judge_bp.route('/api/drive-judge', methods=['POST'])
def analyze_drive_judge_route():
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "교통사고 설명이 필요합니다."}), 400

    try:
        answer = analyze_drive_judge(query)
        return jsonify({"answer": answer}), 200
    except Exception as e:
        print(f"Error in analyze_drive_judge_route: {str(e)}")  # 에러 로깅
        return jsonify({"error": f"응답 생성 중 오류가 발생했습니다: {str(e)}"}), 500
