import requests
from bs4 import SoupStrainer, BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma

from bs4_search.constants import BS4_DOCUMENTATION_URL, CHROMA_PERSIST_DIRECTORY, CHROMA_COLLECTION_NAME, \
    EMBEDDING_FUNCTION


# Extract metadata from a section
def extract_metadata(section):
    # Extract the section ID
    results = {"section-id": section.attrs["id"]}

    # Extract the section title
    section_title = section.find("h1")
    if section_title:
        results["section-title"] = section_title.get_text()

    return results


# Split the section text into Documents
def split_into_documents(texts, metadatas):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,  # Set the desired chunk size (number of characters)
        chunk_overlap=50,  # Set the desired overlap between chunks (number of characters)
        length_function=len,  # Specify the function to measure the length of the text
        is_separator_regex=False  # Specify whether the separators are regular expressions
    )
    return splitter.create_documents(texts, metadatas)


# Extract the documentation sections
def extract_documentation_sections(response):
    strainer = SoupStrainer("section")
    soup = BeautifulSoup(response.content, "html.parser", parse_only=strainer)
    return soup.find_all("section")


def create_documents(documentation_sections):
    sections = []
    metadatas = []

    for section in documentation_sections:
        metadatas.append(extract_metadata(section))
        sections.append(section.get_text())

    return split_into_documents(sections, metadatas)


def add_documents_to_store(documents):
    vector_store = Chroma(persist_directory=CHROMA_PERSIST_DIRECTORY, collection_name=CHROMA_COLLECTION_NAME,
                          embedding_function=EMBEDDING_FUNCTION)
    vector_store.add_documents(documents)
    vector_store.persist()


def ingest_documents():
    response = requests.get(BS4_DOCUMENTATION_URL)
    sections = extract_documentation_sections(response)
    documents = create_documents(sections)
    add_documents_to_store(documents)


if __name__ == "__main__":
    ingest_documents()
