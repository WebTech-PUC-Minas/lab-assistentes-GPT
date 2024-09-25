<!-- Exemplo de uso do template: https://github.com/kspencerl/lab-springboot-basic-api -->

# Lab-assistentes

<!--Breve descrição do projeto aqui -->

## Tecnologias utilizadas

<!-- Link com os badges para inserir abaixo https://devicon.dev/ -->

<div style="display: flex; gap: 10px;">
  <img width="50px" src="https://avatars.githubusercontent.com/u/126733545?v=4">   
  <img width="50px" src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/python/python-original.svg"/>  
</div>

## Onde Aplicar

Este projeto pode ser aplicado nas seguintes situações:

- Assistentes virtuais, para atendiementos simples.
- Assitentes pessoais especializadas em algo.

# Sumário

* [Instalações](#instalações)
  * [Pré-Requisitos](#pré-requisitos)
  * [Configuração de Ambiente](#configuração-de-ambiente)
  * [Base Code](#Base-Code)
  * [Arquivos](#Arquivos)
* [Contato](#Contato)

## Instalações

Siga com precisão as orientações de configuração do ambiente para assegurar eficácia consistente no desenvolvimento do projeto.

### Pré-Requisitos

- **[Pyton](https://www.python.org/downloads/)**

### Configuração de Ambiente
O comando pip para instalar as bibliotecas nessesarias para este projeto é:

```bash
pip install langchain langchain_chroma langchain_core langchain_text_splitters langchain_openai langchain_community pypdf
```

> [!IMPORTANT]
> Dependendo do formato de arquivo pode ser que o loader se altere

## Base Code

### Base vector

Base vector é a parte usada para a captura e divisao dos documentos que seram usados no chat, este codigo é dividido nas seguintes partes:

###### Importação das bibliotecas

Este import tem a função de carregar o documento, podendo ser mudado dependendo do formato do arquivo.
```python
from langchain_community.document_loaders import PyPDFLoader
```

Biblioteca utilizada para a criação dos embeddings, feito para converter os textos em vetores de embeddings que podem ser usados para busca e análise.
```python
from langchain_openai import OpenAIEmbeddings
```

Usado para a divisão dos documentos em partes menores, para que a pesquisa possa ser mais eficiente.
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

Biblioteca para a estruturação dos arquivos estarem no modelo do Chroma.
```python
from langchain_chroma import Chroma
```

###### Carga do documento

```python
path_vetorDb = "path da saida"
loader = PyPDFLoader("path do arquivo", extract_images=False)
docs = loader.load()
```

###### Criação dos Chunks

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
```

###### Passagem para vector store

```python
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key=openai_api_key), persist_directory=path_vetorDb)
```

> [!NOTE]
> Se for feito dessa forma o Vector fica armazenado no disco, fazendo assim que o codigo tenha que ser executado apenas uma vez.

### Base Code (parte do Chat)

Este codigo é o chat propriamente dito, com todas as partes do desenvolvimento, que possue as seguintes partes:

###### Importação das bibliotecas

Biblioteca para criar uma cadeia para combinar documentos em um único documento ou resposta.

```python
from langchain.chains.combine_documents import create_stuff_documents_chain
```

Biblioteca para criar uma cadeia para recuperar documentos relevantes com base em uma consulta.

```python
from langchain.chains import create_retrieval_chain
```

Biblioteca para criar um recuperador de documentos que leva em consideração o histórico de interações

```python
from langchain.chains import create_history_aware_retriever
```

Biblioteca para a interface para a integração com a base de dados Chroma

```python
from langchain_chroma import Chroma
```

Biblioteca para criar templates para prompts de chat

```python
from langchain_core.prompts import ChatPromptTemplate
```

Biblioteca para placeholder para mensagens em um prompt de chat
```python
from langchain_core.prompts import MessagesPlaceholder
```

Biblioteca para definir tipos de mensagens para interações de chat (mensagens de IA e mensagens de humanos)

```python
from langchain_core.messages import AIMessage, HumanMessage
```

Biblioteca utilizada para a criação dos embeddings, feito paca conevrter os textos em vetores de embeddings que podem ser usados para busca e análise.
```python
from langchain_openai import OpenAIEmbeddings
```

Biblioteca para interface para usar modelos de chat da OpenAI
```python
from langchain_openai import ChatOpenAI
```

###### llm config:

```python
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0, # A temperatura varia de 0 ate 1, 0 mais padronizado, 1 mais aleatório
    max_tokens=None,
    api_key=openai_api_key,
    # timeout=None,
    # max_retries=2
    )
```

###### Carga dos documentos:

```python
path_vetorDb = "Pyton/Data/vetorDB"
vectorstore = Chroma(persist_directory=path_vetorDb, embedding_function=OpenAIEmbeddings(api_key=openai_api_key))
retriever = vectorstore.as_retriever()
```

###### Prompt de configuração:

```python
system_prompt = (
    ""
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
```

###### Criação das chains de pesquisa:

Chain que utiliza o retriever para recuperar os dados, question_answer_chain para combinar os dados em uma resposta
```python
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
```

###### Crição do retorno para o historico:

```python
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
```

###### Acesso do chat para as partes anteriores:

```python
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
```

###### Chat:

```python
chat_history = []

while True:
  # Pergunta ao usuário
  question = input("Digite sua pergunta: ")

  # Processa a pergunta e gera a resposta da IA
  ai_msg = rag_chain.invoke({"input": question, "chat_history": chat_history})
  chat_history.extend([HumanMessage(content=question), AIMessage(content=ai_msg["answer"])])

  # Imprime a resposta da IA
  print(ai_msg["answer"] + "\n")
```

---

## Arquivos

#### Aeds-II

Sera uma assistente especializada na diciplina de AEDS-II, com conhecimento dos codigos propostos pelo professor Doutor Max do Val Machado

---

#### Asistente de intenet

Seria um asistente basica para solução de problemas relacionados a modem e intenet no geral

---


#### Gestor de crises

É uma inteligencia artificial especializada para lidar com crises e auxiliar funcionarios com a tenção

---

> [!IMPORTANT]
> NENHUMAS DAS IAs CONTEM EXEMPLO DE DADOS ENTÃO ALGUMAS FUNCIONALIDADES ESTÃO RESTRITAS

## Contato

Emails:

Henrique Nahim: hsnahim@gmail.com

Claudio Dias: claudioangontijo@gmail.com

Gabriel: gabrielhspereira2020@gmail.com

Github:

Henrique Nahim: [github.com/hsnahim](github.com/hsnahim)

Claudio Dias: [github.com/claudiogpt](github.com/claudiogpt)

Gabriel: [github.com/Gab-HSP](https://github.com/Gab-HSP)
