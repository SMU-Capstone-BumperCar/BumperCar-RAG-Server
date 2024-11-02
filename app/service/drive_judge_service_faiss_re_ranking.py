from langchain_community.vectorstores import FAISS
from langchain_upstage import UpstageEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from config import GEMINI_API_KEY, DRIVE_JUDGE_PROMPT_TEMPLATE, UPSTAGE_API_KEY, COHERE_API_KEY

# Upstage 임베딩 모델 설정
us_model = UpstageEmbeddings(api_key=UPSTAGE_API_KEY, model="solar-embedding-1-large")

# FAISS DB 경로 설정
faiss_db_path = "./app/faiss_db/drive_judge_dataset_faiss"

# 기존 FAISS 벡터 스토어 로드
try:
    faiss_db = FAISS.load_local(faiss_db_path, embeddings=us_model, allow_dangerous_deserialization=True)
    print("기존 FAISS 데이터베이스 로드 성공.")
except Exception as e:
    print("FAISS 데이터베이스를 로드할 수 없습니다:", str(e))
    raise

# 검색 기능 설정
retriever = faiss_db.as_retriever(search_type='similarity', search_kwargs={'k': 5})

# 리랭킹을 위한 압축 리트리버 설정
compressor = CohereRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

# 프롬프트 템플릿 정의
prompt = ChatPromptTemplate.from_template(DRIVE_JUDGE_PROMPT_TEMPLATE)

# 언어 모델 설정
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, api_key=GEMINI_API_KEY)

# 대화 기록 관리
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def merge_context(pages, conversation_history):
    """검색된 페이지와 대화 기록을 병합."""
    merged_pages = "\n\n".join(page.page_content for page in pages)
    return f"{conversation_history}\n\n{merged_pages}"

def analyze_drive_judge(query):
    # 이전 대화 내용 로드
    conversation_history = "\n".join([msg.content for msg in memory.load_memory_variables({}).get("chat_history", [])])

    # 리랭킹된 문서를 사용하여 실행 체인 설정
    compressed_docs = compression_retriever.get_relevant_documents(query)
    
    chain = (
        {
            "query": RunnablePassthrough(),
            "context": lambda _: merge_context(compressed_docs, conversation_history)
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # 응답 생성 후 대화 내용 저장
    try:
        answer = chain.invoke(query)
        memory.save_context({"query": query}, {"response": answer})
    except Exception as e:
        print(f"Error generating response: {str(e)}")  # 에러 로깅
        answer = "응답 생성 중 오류가 발생했습니다."

    return answer
