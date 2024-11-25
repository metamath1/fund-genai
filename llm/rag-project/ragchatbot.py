import random, os, time
from io import BytesIO
import streamlit as st

from dotenv import load_dotenv
import faiss

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, KonlpyTextSplitter


from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_core.prompts import MessagesPlaceholder
# from langchain_core.messages import AIMessage, HumanMessage

# streamlit을 사용하여 챗봇을 만드는 튜토리얼 코드
# https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps

# 이 스크립트 테스트 환경 
# windows@metamath st2 conda 환경에서 테스트 됨

#############################################################################
# 시스템 프롬프트 탬플릿 
# 이 프롬프트를 써서 pdf파일을 업로드하고 같이 이야기 할 수 있냐고 물으면
# 아래처럼 가능하다고 대답하고 왼쪽 패널에서 파일을 업로드하라고 이야기함
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "You can have a conversation based on a PDF file."
    "When a user asks to chat using a PDF file, reply in Korean like this example"
    "-----------------"
    "네 가능합니다! 우선 함께 대화 나눌 pdf를 업로드 해주세요. 업로드는 왼쪽 패널에서 할 수 있습니다."
    "-----------------"

    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# 메모리를 가지는 리트리버를 위한 부분
# 채팅이 반복되면서 마지막에 사용자의 질문이 들어오면
# 그 들어온 질문을 
# 과거 대화 이력을 참고하여 그 이력을 모두 포함 할 수 있는 다시말해
# 과거 대화 맥락을 파악할 수 있는 질문으로 재생성하라는 프롬프트
# 이렇게 맥락을 포함한 질문을 재생성해서 재생성된 질문과
# 관련이 있는 문서 청크들을 검색해올 목적으로 사용됨 

# 원래 프롬프트
# contextualize_q_system_prompt = """Given a chat history and the latest user question \
# which might reference context in the chat history, formulate a standalone question \
# which can be understood without the chat history. Do NOT answer the question, \
# just reformulate it if needed and otherwise return it as is."""
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history.

!!!CRITICAL INSTRUCTION!!!
YOUR ONLY TASK IS TO REFORMULATE THE QUESTION IF NEEDED.

- DO NOT provide any answers or explanations
- DO NOT offer assistance or suggestions
- DO NOT engage in discussion
- ONLY output either:
    1. The original question (if it's already standalone) OR
    2. A reformulated standalone version of the question

Response MUST BE a question.
"""


contextualize_q_prompt_tpl = ChatPromptTemplate.from_messages(
    [
        # 시스템 프롬프트
        ("system", contextualize_q_system_prompt),

        # 지금까지 채팅 히스토리
        MessagesPlaceholder("chat_history"),

        # 새로 주어진 사용자의 요청(질문)
        ("human", "{input}"),
    ]
)
#############################################################################

# 업로드된 pdf_docs에 있는 pdf파일을 PyPDFLoader를 이용해서 문서를
# 랭체인 Document 객체 형태로 로딩하는 함수
def get_pdf_text(pdf_docs):
    pages = []
    
    # 업로드된 파일 하나하나에 대해서 for루프를 돌면서...
    for uploaded_pdf in pdf_docs: # uploaded_pdf는 streamlit의 UploadedFile 객체
        print(f"{uploaded_pdf.name}로 부터 텍스트 추출 시작")
        
        # streamlit은 업로드한 파일을 메모리에만 저장하므로 디스크로 저장하고
        # langchain PyPDFLoader로 읽음
        save_path = os.path.join("uploaded_files", uploaded_pdf.name)

        with open(save_path, "wb") as f:
            f.write(uploaded_pdf.read())
            print(f"{save_path}에 파일 저장 완료")

        # 저장된 파일을 디스크로 부터 로딩
        doc_loader = PyPDFLoader(save_path)
        pages += doc_loader.load_and_split()

        print(f"{uploaded_pdf.name}로 부터 텍스트 추출 완료")

    return pages
   
# 위 get_pdf_text()가 로딩된 pdf 내용을 반환하면 그것을 입력받아
# 다시 chunk_size로 자르는 함수
def get_text_chunks(langchain_docs):

    # text_splitter = RecursiveCharacterTextSplitter(
    text_splitter = KonlpyTextSplitter(
        # separator='.',
        chunk_size=120,  # 청크(문서의 조각)의 크기를 지정
        chunk_overlap=0, # 잘려진 청크가 얼마나 겹칠지를 지정
        length_function=len # 문서를 자를때 문서의 길이를 len()으로 판단하라고 지정 
    )

    # text_splitter로 문서를 자름 
    chunks = text_splitter.split_documents(langchain_docs)
    if len(chunks) > 0:
        print("전체 pdf 청킹 완료")

    return chunks

# 위 get_text_chunks()함수가 잘라준 작은 문서의 조각들을
# 벡터로 변환하여 데이터 베이스에 저장하는 함수
def get_vectorstore(doc_chunks):
    # OpenAI에서 제공하는 유료 모델을 사용할 수 도 있고
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # 허깅페이스에서 호스팅하는 BERT 모델을 다운받아 로컬에서 직접
    # 임베딩할 수도 있음
    # embeddings = HuggingFaceEmbeddings(
    #     model_name='jhgan/ko-sbert-nli',
    #     model_kwargs={'device':'cpu'},
    # )
    
    # pdf 파일이 업로드되지 않아 문서 청크를 저장한 리스트가 빈 리스트라면 
    if not doc_chunks:
        print("doc_chunks가 비어있습니다. 빈 벡터 DB를 생성합니다.")
        # 빈 벡터 DB를 생성
        vectorstore = FAISS(
            embedding_function=embeddings, 
            index=faiss.IndexFlatL2(len(embeddings.embed_query("hello world"))),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
    else:
        # doc_chunks가 비어있지 않으면 정상적으로 벡터 DB 생성
        vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    
    print("벡터 스토어 생성 완료")
    print(f"총 저장된 벡터 수: {vectorstore.index.ntotal}")

    return vectorstore

# 위에서 정의한 각 단계를 담당하는 함수를 순서대로 실행하면서
# 대화의 히스토리를 반영하여 질문과 관련된 문서 청크를 검색해서 
def create_history_aware_rag_chain(pdf_docs):
    ################################################################
    # load documents
    docs_list_from_pdfs = get_pdf_text(pdf_docs)
    ################################################################
    

    ################################################################
    # text splitter
    doc_chunks = get_text_chunks(docs_list_from_pdfs)
    ################################################################


    ################################################################
    # create vector store
    vectorstore = get_vectorstore(doc_chunks)
    ################################################################


    ################################################################
    # create chain

    # 벡터 DB를 체인에서 사용하는 리트리버로 변경하고
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 10} # 검색 청크 개수
    )

    # contextualize_q_prompt_tpl에 의해 
    # 과거 대화 맥락을 바탕으로 사용자의 질문을 재생성하는 질문을
    # 전달하는 llm이 만들고 그 질문과 관련있는 문서 청크를 retriever가
    # 검색하는 역할을 하는 리트리버를 생성
    # https://python.langchain.com/docs/tutorials/qa_chat_history/
    st.session_state.history_aware_retriever = create_history_aware_retriever(
        llm, 
        retriever, contextualize_q_prompt_tpl
    )
    
    # 최종적으로 사용자의 질문을 처리하는 프롬프트와 체인을 만듬
    qa_prompt_tpl = ChatPromptTemplate.from_messages(
        [
            # 여기 system_prompt안에 context 변수가 있는데(위 system_prompt 정의 참고)
            # 이 context 변수에는 history_aware_retriever가 새롭게 질문을 구성하고
            # 검색한 청크들로 채워지게 됨
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm_mini, qa_prompt_tpl)

    # 마지막으로 질문에 답변하는 체인과 대화 맥락을 반영하여 문서 청크를 검색하는
    # 리트리버를 연결하여 최종적인 rag_chain을 만듬
    # 만들어진 체인을 streamlit 세션 변수에 대입하여 전역변수처럼 사용할 수 있게 준비
    st.session_state.rag_chain = create_retrieval_chain(
        st.session_state.history_aware_retriever, 
        question_answer_chain
    )

# 간단하게 가상으로 답변을 되돌리는 함수
# llm으로 제대로된 체인을 구성하기 전에 전체 앱을 테스트해볼 용도로 만들어짐
def get_random_msg():
    messages = [
        "날씨가 참 좋군요!",
        "어제 혹시 즐거운 쇼핑을 하셨나요?",
        "오늘은 좋은 일만 있었으면 좋겠네요.",
        "언제 퇴근하세요?",
        "혹시 어디 아프신가요?"
    ]

    response = random.choices(messages, k=1)[0]
        
    return response

# 사용자로 부터 입력이 들어오면 처리하는 함수
def handle_user_input(user_input):
    # 먼저 디버깅 목적으로 history_aware_retriever 체인을 invoke해서
    # history_aware_retriever가 되돌리는 청크를 화면서 출력하여 확인
    response_foo = st.session_state.history_aware_retriever.invoke(
        {
            "input": user_input,
            "chat_history": st.session_state.messages            
        }
    )
    print("\nhistory_aware_retriever의 응답")
    print("CHUNKS--------------------")
    for i, chunk in enumerate(response_foo):
        print(f"{i}번째 청크: ", chunk.page_content)
        print()
    print("---------------------------")

    # 사용자 입력에 답변하는 체인을 작동 시킴
    response = st.session_state.rag_chain.invoke(
        {
            "input": user_input,
            "chat_history": st.session_state.messages
        }
    )
    # 실제 llm체인의 응답 response는 다음처럼 생겼음
    # {
    #     'input': 방금 받은 질문
    #     'chat_history': [
    #         {'role':'human', 'content':'안녕하세요.'},
    #         {'role':'assistant', 'content':'블라블라...'},
    #         # 이런식으로 위에 최근 질답까지 포함한 총 질답의 리스트
    #     ],
    #     'answer': 질문에 대한 답변
    # }

    return response['answer']

def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


if __name__ == '__main__':
    # 환경 변수 로딩
    load_dotenv()
    # print(os.getenv("OPENAI_API_KEY"))
    # print(os.getenv("HUGGINGFACEHUB_API_TOKEN"))

    # 업로드된 pdf 내용을 저장할 리스트
    # pdf 내용은 랭체인의 Document() 형식으로 로딩됨
    docs_list_from_pdfs = []

    #################################################################################
    # LLM 정의
    # rag 컨텍스트와 함께 사용자의 질문에 답변하기 위해 사용
    llm_mini = ChatOpenAI(
        temperature = 0.0,
        model='gpt-4o-mini', 
        # 이 옵션을 주면 콘솔에 실시간으로 메세지가 뿌려짐
        # 두개의 체인이 돌아가는데 각 체인의 내부 출력이 화면에 뿌려지게 할 수 있음
        # streaming=True,
        # callbacks=[StreamingStdOutCallbackHandler()]
    )

    # 여기서는 chat history를 반영한 standalone 질문을 생성하기 위해 사용
    llm = ChatOpenAI(
        temperature = 0.0,
        model='gpt-4o', 
        streaming=True,
        # 이 옵션을 주면 콘솔에 실시간으로 메세지가 뿌려짐
        # 두개의 체인이 돌아가는데 각 체인의 내부 출력이 화면에 뿌려지게 할 수 있음
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    #################################################################################



    ##########################################################################
    # 사이드바 정의 영역
    ##########################################################################
    with st.sidebar:
        st.subheader("업로드된 PDF 문서")

        # pdf_docs: list of pdf files
        pdf_docs = st.file_uploader(
            "PDF 파일들을 업로드하고 'Process' 버튼을 누르세요.",
            accept_multiple_files=True # 여러 파일 업로드
        )
        
        #######################################################################
        # 업로드된 pdf 파일 전처리 부분
        if st.button("Process"): # 사용자가 버튼을 눌렀으면 매번 새로운 체인 생성
            with st.spinner("Processing..."):
                # 현재 업로드된 pdf파일 (pdf_docs)를 넘기면서 체인 생성
                create_history_aware_rag_chain(pdf_docs)
        else: 
            # 세션 변수에 rag_chain이라는 항목이 없으면 최초로 체인을 생성
            # 이때는 process 버튼을 누르지 않았기 때문에
            # pdf파일을 처리하지 않고 그냥 체인만 만듬 
            if "rag_chain" not in st.session_state:
                create_history_aware_rag_chain(pdf_docs)
        #######################################################################
        
        if st.button("chat_history"):
            print('-------chat_history---------')
            print(st.session_state.messages)
            print('----------------------------')
        # debugging pdf_docs 
        # UploadedFile 객체
        # 업로드된 객체를 화면에 뿌림 
        # print("pdf_docs", pdf_docs)
        # print("docs_list_from_pdfs", docs_list_from_pdfs)
    ##########################################################################
    # 사이드바 정의 영역
    ##########################################################################

    ##########################################################################
    # 메인화면 정의 영역
    ##########################################################################
    st.title("🔎 LangChain - RAG")

    # st.session_state['messages'] 는 이 앱이 메세지를 기록하기 위한 버퍼
    # 기록되는 레코드 형식은 {'role':'', 'content':''} 식이고
    # 처음에 기록된게 없다면 빈 리스트로 초기화
    if "messages" not in st.session_state:
        st.session_state["messages"] = [ ]

    # st.chat_message는 화면에 뿌리는 용도의 버퍼
    # st.chat_message(role)로 write를 하면 가각 롤에 맞게 화면에 뿌려짐
    for msg in st.session_state.messages:
       st.chat_message(msg["role"]).write(msg["content"])

    # st.chat_input은 화면 아래 나타나는 유저 인풋을 받는 입력창
    if prompt := st.chat_input(placeholder="어떤 질문이든 해보세요."):
        # 유저 인풋을 메세지 히스토리에 저장하고
        st.session_state.messages.append({"role": "user", "content": prompt})
        # 바로 화면에 출력
        st.chat_message("user").write(prompt)

        # 여기서 체인 돌리기
        # response = get_random_msg() # 시험 삼아 무작위 응답 받아오기 
        response = handle_user_input(prompt)

        with st.chat_message("assistant"):
            # llm의 응답을 메세지 히스토리에 저장하고 
            st.session_state.messages.append({"role": "assistant", "content": response})
            # 출력, 여기서는 with 문안에 있기 때문에 st.write()만 해도
            # st.chat_message("assistant").write()로 작동함
            st.write_stream(response_generator(response))
    ##########################################################################
    # 메인화면 정의 영역        
    ##########################################################################