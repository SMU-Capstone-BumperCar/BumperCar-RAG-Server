import os
import sys
import re
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_upstage import UpstageEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# config 파일 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config import UPSTAGE_API_KEY

# 텍스트 파일 로드
loader = TextLoader("./app/data/car_to_car_sample.txt", encoding="utf-8")
documents = loader.load()  # 텍스트 파일의 내용을 가져오기

# 텍스트 청킹 설정 (패턴 기준 대분류 청킹)
pattern = r'(차\d+-\d+.+?)(?=(차\d+-\d+|$))'
text = documents[0].page_content  # documents 리스트에서 텍스트 추출

# 패턴을 사용하여 대분류 청크 단위로 분할
large_chunks = [match[0] if isinstance(match, tuple) else match for match in re.findall(pattern, text, re.DOTALL)]

# RecursiveCharacterTextSplitter를 통해 추가 청킹 설정
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=100,
    separators=["\n\n", "\n", " "]
)

# 대분류 청크 내에서 추가 청킹하여 세분화된 청크 리스트 생성
final_chunks = []
for chunk in large_chunks:
    split_chunks = text_splitter.split_text(chunk)
    final_chunks.extend(split_chunks)

# 청크 확인: 첫 5개의 청크 내용과 길이 출력
print("=== 청크 예시 (첫 5개) ===")
for i, chunk in enumerate(final_chunks[:7]):
    print(f"청크 {i+1} (길이: {len(chunk)}):\n{chunk}\n{'-'*50}\n")

# Upstage 임베딩 모델 설정
embedding_model = UpstageEmbeddings(api_key=UPSTAGE_API_KEY, model="solar-embedding-1-large")

# FAISS DB 생성 또는 불러오기
faiss_db_path = "./app/vector_databases/faiss_db/drive_judge_dataset_faiss"
try:
    # 기존 데이터베이스 불러오기
    faiss_db = FAISS.load_local(faiss_db_path, embeddings=embedding_model)
    print("\n기존 FAISS 데이터베이스 불러오기 완료.")
except Exception as e:
    # 데이터베이스가 없으면 새로 생성
    print("\n기존 데이터베이스를 찾을 수 없어 새 데이터베이스를 생성합니다.")
    faiss_db = FAISS.from_texts(final_chunks, embedding_model)
    faiss_db.save_local(faiss_db_path)
    print("임베딩 완료 및 FAISS DB에 저장되었습니다.")
