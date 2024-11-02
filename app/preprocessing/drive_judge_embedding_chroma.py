import os
import sys

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_upstage import UpstageEmbeddings

# config.py 파일이 위치한 경로를 기준으로 상위 경로를 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config import UPSTAGE_API_KEY

loader = PyPDFLoader("./app/data/drive_judge.pdf")
documents = loader.load()  

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    length_function=len,
)

# PDF 내용을 청킹한 텍스트 리스트로 변환
texts = text_splitter.split_documents(documents)
doc_list = [doc.page_content for doc in texts]

# Upstage 임베딩 모델 설정
embedding_model = UpstageEmbeddings(api_key=UPSTAGE_API_KEY, model="solar-embedding-1-large")

# 크로마 DB에 저장 또는 생성 후 저장
chroma_db_path = "./app/vector_databases/chroma_db/drive_judge_dataset"
try:
    # 기존 데이터베이스 불러오기
    chroma_db = Chroma(persist_directory=chroma_db_path, embedding_function=embedding_model)
    print("\n기존 데이터베이스 불러오기 완료.")
except Exception as e:
    # 데이터베이스가 없으면 새로 생성
    print("\n기존 데이터베이스를 찾을 수 없어 새 데이터베이스를 생성합니다.")
    chroma_db = Chroma.from_documents(doc_list, embedding_model, persist_directory=chroma_db_path)

# 새 문서 추가
# chroma_db.add_documents(doc_list)
# print("임베딩 완료 및 크로마DB에 저장되었습니다.")
