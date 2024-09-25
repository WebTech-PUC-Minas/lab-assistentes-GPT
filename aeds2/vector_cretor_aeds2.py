from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
# Caso não seja possivel colocar a chave nas variaveis de ambiente incira manualmente aqui
# Senao deixe vazio
openai_api_key=""

path_vetorDb = "aeds2/vetorDB"
loader = TextLoader("aeds2/arquivos_unidos.txt")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key=openai_api_key), persist_directory=path_vetorDb)