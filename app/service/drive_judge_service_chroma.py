from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_upstage import UpstageEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from config import GEMINI_API_KEY, DRIVE_JUDGE_PROMPT_TEMPLATE, UPSTAGE_API_KEY

# Upstage 임베딩 모델 설정
us_model = UpstageEmbeddings(api_key=UPSTAGE_API_KEY, model="solar-embedding-1-large")

# Chroma DB 설정
chroma_db = Chroma(persist_directory="app/chroma_db/drive_judge_dataset_2", embedding_function=us_model)
retriever = chroma_db.as_retriever(search_type='similarity', search_kwargs={'k': 5})

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

    # 실행 체인 설정
    chain = (
        {
            "query": RunnablePassthrough(),
            "context": retriever | (lambda pages: merge_context(pages, conversation_history))
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
