import os
import sys

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_upstage import UpstageEmbeddings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config import UPSTAGE_API_KEY

loader = PyPDFLoader("./app/data/drive_judge.pdf")
documents = loader.load() 

# 텍스트 청킹 (chunking) 설정
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

# PDF 내용을 청킹한 텍스트 리스트로 변환
texts = text_splitter.split_documents(documents)
doc_list = [doc.page_content for doc in texts]

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
    faiss_db = FAISS.from_texts(doc_list, embedding_model)
    faiss_db.save_local(faiss_db_path)
    print("임베딩 완료 및 FAISS DB에 저장되었습니다.")
