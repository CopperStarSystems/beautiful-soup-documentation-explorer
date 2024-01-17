from operator import itemgetter
from typing import Optional, List, Dict, Sequence

from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnableBranch, Runnable, \
    RunnableMap
from langchain_openai import ChatOpenAI

from bs4_search.constants import CHROMA_PERSIST_DIRECTORY, CHROMA_COLLECTION_NAME, EMBEDDING_FUNCTION

__vector_store__ = Chroma(persist_directory=CHROMA_PERSIST_DIRECTORY, collection_name=CHROMA_COLLECTION_NAME,
                          embedding_function=EMBEDDING_FUNCTION)

# retriever = vectorstore.as_retriever()

__RESPONSE_TEMPLATE__ = """\
You are an expert programmer and problem-solver, tasked with answering any question \
about Beautiful Soup 4.

Generate a comprehensive and informative answer of 80 words or less for the \
given question based solely on the provided context. You must \
only use information from the provided context. Use an unbiased and \
journalistic tone. Combine search results together into a coherent answer. Do not \
repeat text.

You should use bullet points in your answer for readability. Put citations where they apply
rather than putting them all at the end.

If there is nothing in the context relevant to the question at hand, just say "Hmm, \
I'm not sure." Don't try to make up an answer.

Anything between the following `context`  html blocks is retrieved from a knowledge \
bank, not part of the conversation with the user. 

<context>
    {context} 
<context/>

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \
not sure." Don't try to make up an answer. Anything between the preceding 'context' \
html blocks is retrieved from a knowledge bank, not part of the conversation with the \
user.\
"""

__REPHRASE_TEMPLATE__ = """\
Given the following conversation and a follow up question, rephrase the follow up \
question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:
"""


class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]]


def __get_retriever__() -> BaseRetriever:
    return __vector_store__.as_retriever()


def __create_retriever_chain__(
        llm: BaseLanguageModel, retriever: BaseRetriever
) -> Runnable:
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(__REPHRASE_TEMPLATE__)
    condense_question_chain = (
            CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()
    ).with_config(
        run_name="CondenseQuestion",
    )
    conversation_chain = condense_question_chain | retriever
    return RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),
            conversation_chain.with_config(run_name="RetrievalChainWithHistory"),
        ),
        (
                RunnableLambda(itemgetter("question")).with_config(
                    run_name="Itemgetter:question"
                )
                | retriever
        ).with_config(run_name="RetrievalChainWithNoHistory"),
    ).with_config(run_name="RouteDependingOnChatHistory")


def __format_docs__(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def __serialize_history__(request: ChatRequest):
    chat_history = request["chat_history"] or []
    converted_chat_history = []
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    return converted_chat_history


def __create_chain__(
        llm: BaseLanguageModel,
        retriever: BaseRetriever,
) -> Runnable:
    retriever_chain = __create_retriever_chain__(
        llm,
        retriever,
    ).with_config(run_name="FindDocs")
    _context = RunnableMap(
        {
            "context": retriever_chain | __format_docs__,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
    ).with_config(run_name="RetrieveDocs")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", __RESPONSE_TEMPLATE__),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    response_synthesizer = (prompt | llm | StrOutputParser()).with_config(
        run_name="GenerateResponse",
    )
    return (
            {
                "question": RunnableLambda(itemgetter("question")).with_config(
                    run_name="Itemgetter:question"
                ),
                "chat_history": RunnableLambda(__serialize_history__).with_config(
                    run_name="SerializeHistory"
                ),
            }
            | _context
            | response_synthesizer
    )


retriever = __get_retriever__()
llm = ChatOpenAI(
    model="gpt-3.5-turbo-16k",
    streaming=True,
    temperature=0,
)
answer_chain = __create_chain__(
    llm,
    retriever,
)

chain = answer_chain
#
#
#
#
#
# prompt = ChatPromptTemplate.from_template(template)
#
# # LLM
# model = ChatOpenAI()
#
# # RAG chain
# chain = (
#         RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
#         | prompt
#         | model
#         | StrOutputParser()
# )
#
#
# # Add typing for input
# class Question(BaseModel):
#     __root__: str
#
#
# chain = chain.with_types(input_type=Question)
