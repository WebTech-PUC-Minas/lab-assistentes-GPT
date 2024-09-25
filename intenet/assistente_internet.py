# pip install langchain langchain_chroma langchain_core langchain_text_splitters langchain_openai langchain_community pypdf
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

# Caso não seja possivel colocar a chave nas variaveis de ambiente incira manualmente aqui
# Senao deixe vazio
openai_api_key=""

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3, # A temperatura varia de 0 ate 1
    max_tokens=None,
    api_key=openai_api_key,
    # timeout=None,
    # max_retries=2
    )
path_vetorDb = "intenet/vetorDB"
vectorstore = Chroma(persist_directory=path_vetorDb, embedding_function=OpenAIEmbeddings(api_key=openai_api_key))
retriever = vectorstore.as_retriever()

system_prompt = (
    "Você é um assitente de internet que deve auxiliar os usuários a resolver problemas relacionados a internet."
    "inicialmente voce devera diagnosticar o problema e em seguida fornecer a solução."
    "não existe nessecidade de ser extremamente técnico, mas deve ser claro e preciso."
    "seja mais humano siga um passo a passo e tente resolver o problema do usuário nunca avançando mais de um passo por vez"
    "lembre-se caso não seja capas de resolver o problema, você deve encaminhar o usuário para o suporte técnico localizado no numero (99)999."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

contextualize_q_system_prompt = (
    "Você é um assitente de internet que deve auxiliar os usuários a resolver problemas relacionados a internet."
    "inicialmente voce devera diagnosticar o problema e em seguida fornecer a solução."
    "lembre-se caso não seja capas de resolver o problema, você deve encaminhar o usuário para o suporte técnico localizado no numero (99)999."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# Vetor do chat
chat_history = []

while True:
  # Pergunta ao usuário
  question = input("Digite sua pergunta: ")

  # Processa a pergunta e gera a resposta da IA
  ai_msg = rag_chain.invoke({"input": question, "chat_history": chat_history})
  chat_history.extend([HumanMessage(content=question), AIMessage(content=ai_msg["answer"])])

  # Imprime a resposta da IA
  print(ai_msg["answer"] + "\n")
