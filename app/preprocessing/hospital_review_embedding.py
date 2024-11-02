import os
from dotenv import load_dotenv
import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_upstage import UpstageEmbeddings

load_dotenv()
upstage_api_key = os.getenv("UPSTAGE_API_KEY")
if not upstage_api_key:
    raise ValueError("UPSTAGE_API_KEY가 설정되지 않았습니다.")

df = pd.read_csv("app/data/hospital_review_dataset.csv")
print("\n\nhospital_review_dataset 파일 읽기 완료")

# 데이터프레임의 각 행을 Document 객체로 변환
documents = [
    Document(page_content=f"{row['hospital_name']},{row['content']},{row['date']},{row['revisit']}", metadata={}) for _, row in df.iterrows()
]
print(f"Document 개수: {len(documents)}")

# Upstage Embeddings 설정
us_model = UpstageEmbeddings(
    api_key=upstage_api_key,
    model="solar-embedding-1-large"
)
print("임베딩 모델 설정 완료")

# Chroma DB에 배치로 삽입하는 함수
def add_documents_in_batches(db, documents, batch_size=5461):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        db.add_documents(batch)
        print(f"{min(i + batch_size, len(documents))}개의 문서를 데이터베이스에 추가 완료.")

# Chroma Vector Store 생성 또는 기존 데이터베이스 로드
chroma_db_path = "app/chroma_db/hospital_review_dataset2"


print("\n새 데이터베이스를 생성합니다.")
chroma_db = Chroma(persist_directory=chroma_db_path, embedding_function=us_model)
print("\n기존 데이터베이스 불러오기 완료.")
# add_documents_in_batches(chroma_db, documents)
print("\n임베딩 완료 및 크로마DB에 저장되었습니다.")

# try:
#     # 기존 데이터베이스 불러오기
#     chroma_db = Chroma(persist_directory=chroma_db_path, embedding_function=us_model)
#     print("\n기존 데이터베이스 불러오기 완료.")
# except FileNotFoundError:
#     # 데이터베이스가 없으면 새로 생성
#     chroma_db = Chroma.from_documents(documents[:100], us_model, persist_directory=chroma_db_path)  # 100개 샘플로 생성
#     print("\n기존 데이터베이스를 찾을 수 없어 새 데이터베이스를 생성합니다.")

# add_documents_in_batches(chroma_db, documents)

# # 문서 추가
# add_documents_in_batches(chroma_db, documents)
# print("\n임베딩 완료 및 크로마DB에 저장되었습니다.")
