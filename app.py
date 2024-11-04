from flask import Flask
from app.routes.hospital_review_routes import hospital_review_bp 
from app.routes.drive_judge_routes_faiss import drive_judge_bp  

app = Flask(__name__)

# Blueprint 등록
app.register_blueprint(hospital_review_bp)
app.register_blueprint(drive_judge_bp)

@app.route("/")
def hello():
    return "Hello, Flask!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)