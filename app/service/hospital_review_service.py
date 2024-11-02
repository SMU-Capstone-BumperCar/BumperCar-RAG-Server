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
retriever = chroma_db.as_retriever(search_type='mmr', search_kwargs={'k': 10, 'fetch_k': 40})

# 프롬프트 템플릿 설정
prompt = ChatPromptTemplate.from_template(HOSPITAL_REVIEW_PROMPT_TEMPLATE)

# 언어 모델 설정
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, api_key=GEMINI_API_KEY)

# 병원 리뷰 요약 서비스
def summarize_hospital_review(query):
    def merge_pages(pages):
        merged = "\n\n".join(page.page_content for page in pages)
        return merged

    chain = (
        {"query": RunnablePassthrough(), "context": retriever | merge_pages}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(query)
