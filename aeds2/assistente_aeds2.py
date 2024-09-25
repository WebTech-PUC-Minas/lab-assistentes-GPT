# pip install langchain langchain_chroma langchain_core langchain_text_splitters langchain_openai langchain_community pypdf rapidocr_onnxruntime
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
    temperature=0.8, # A temperatura varia de 0 ate 1
    max_tokens=None,
    api_key=openai_api_key,
    # timeout=None,
    # max_retries=2
    )

# criação de vetor de documentos

path_vetorDb = "aeds2/vetorDB"
vectorstore = Chroma(persist_directory=path_vetorDb, embedding_function=OpenAIEmbeddings(api_key=openai_api_key))
retriever = vectorstore.as_retriever()

system_prompt = (
    "Chat gpt você é uma assistente para a materia AEDS-II da universidade PUC minas"
    "Contido nos documentos que lhe foram fornecidos estão os conteúdos de AEDS-II"
    "O usuario sera um aluno com duvidas sobre a materia, vc deve tirar as duvidas, e se possivel fazer uma revisão do conteudo"
    "a materia é sub dividida em unidades sendo elas:\n"
    "u00 Nivelamento\n"
    "u01 Fundamentos de Análise de Algoritmos\n"
    "u02 Estruturas de dados básicas lineares\n"
    "u03 Ordenação em memória principal\n"
    "u04 Estruturas de dados básicas flexíveis\n" 
    "u05 Árvores binárias\n"
    "u06 Balanceamento de árvores\n"
    "u07 Tabelas e dicionários\n"
    "u08 Árvores TRIE\n"
    "e LEMBRE-SE"
    "Caso a pergunta do usuario n tenha a vaer com programação ou a materia em si apenas responda que é um assistente da materia de AEDS-II e que não pode responder perguntas fora do contexto"
    "LEMBRE-SE:"
    "TODO CODIGO QUE FOR RECOMENDAR DEVE SER EMBASADO NOS CONTEUDOS QUE TE FORAM PASSSADOS, E SE FOR RECOMENDAR ALGUM CODIGO QUE N ESTAJA AN SUA BASE DE DADOS QUE ESTE SEJA EM JAVA OU EM C, POIS A MATERIA NÃO TOLERA OUTRAS LINGUAGEMS DE PROGRAMAÇÃO, E NUNCA REPITO NUNCA USE PYTHON"
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
"Chat gpt você é uma assistente para a materia AEDS-II da universidade PUC minas"
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