import os
import sys
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_upstage import UpstageEmbeddings

# config.py 파일이 위치한 경로를 기준으로 상위 경로를 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config import UPSTAGE_API_KEY

# CSV 파일 읽기
df = pd.read_csv("app/data/hospital_review_dataset.csv")
print("\n\nhospital_review_dataset 파일 읽기 완료")

# 데이터프레임의 각 행을 Document 객체로 변환
documents = [
    Document(
        page_content=row['content'],  # 리뷰 내용만 저장
        metadata={"hospital_name": row['hospital_name']}  # 병원 이름을 metadata로 저장
    ) for _, row in df.iterrows()
]
print(f"Document 개수: {len(documents)}")

# Upstage Embeddings 설정
us_model = UpstageEmbeddings(
    api_key=UPSTAGE_API_KEY,
    model="solar-embedding-1-large"
)
print("임베딩 모델 설정 완료")

# FAISS Vector Store 생성 및 문서 추가 함수 정의
def add_documents_to_faiss(documents, embedding_model):
    # FAISS.from_documents 메서드 사용으로 변경
    faiss_db = FAISS.from_documents(
        documents=documents, 
        embedding=embedding_model  # 여기에 embedding 모델을 전달
    )
    return faiss_db

# FAISS Vector Store 생성
faiss_db = add_documents_to_faiss(documents, us_model)
print("FAISS DB에 문서 임베딩 및 저장 완료") 

# FAISS 저장 경로 지정 및 저장
faiss_db_path = "app/vector_databases/faiss_db/hospital_review_faiss"
faiss_db.save_local(faiss_db_path)
print(f"FAISS 데이터베이스가 '{faiss_db_path}' 경로에 저장되었습니다.")
