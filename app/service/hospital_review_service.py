from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_upstage import UpstageEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import GEMINI_API_KEY, HOSPITAL_REVIEW_PROMPT_TEMPLATE, UPSTAGE_API_KEY

# 임베딩 모델 설정
us_model = UpstageEmbeddings(api_key=UPSTAGE_API_KEY, model="solar-embedding-1-large")

# Chroma DB 설정
chroma_db = Chroma(persist_directory="app/vector_databases/chroma_db/hospital_review_dataset", embedding_function=us_model)
retriever = chroma_db.as_retriever(search_type='mmr', search_kwargs={'k': 200, 'fetch_k': 500})

# 프롬프트 템플릿 설정
prompt = ChatPromptTemplate.from_template(HOSPITAL_REVIEW_PROMPT_TEMPLATE)

# 언어 모델 설정
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, api_key=GEMINI_API_KEY)

# 병원 리뷰 요약 서비스
def summarize_hospital_review(hospital_name):
    # 병원 이름을 포함한 검색 쿼리로 특정 리뷰 가져오기
    search_query = f"{hospital_name}"
    pages = retriever.get_relevant_documents(search_query)
    
    # 병원 이름이 포함된 리뷰만 필터링
    filtered_pages = [page for page in pages if hospital_name in page.page_content]
    
    # 필터링된 리뷰 개수와 내용 출력
    print(f"검색된 리뷰 개수: {len(filtered_pages)}")
    # for i, page in enumerate(filtered_pages, start=1):
    #     print(f"리뷰 {i}: {page.page_content}")
    
    # 필요한 경우 병합하여 모델에 전달
    merged_reviews = "\n\n".join(page.page_content for page in filtered_pages)
    
    # 최종 프롬프트 출력
    final_prompt_content = {
        "query": hospital_name,
        "context": merged_reviews
    }
    print(f"LLM에 전달되는 최종 질문 내용:\n{final_prompt_content}")
    
    # 요약 생성
    chain = (
        {"query": RunnablePassthrough(), "context": RunnablePassthrough() | (lambda x: merged_reviews)}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # 프롬프트에 query와 context를 함께 전달
    return chain.invoke(final_prompt_content)
