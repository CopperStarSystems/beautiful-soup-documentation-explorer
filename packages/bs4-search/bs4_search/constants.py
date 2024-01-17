from langchain_openai import OpenAIEmbeddings

BS4_DOCUMENTATION_URL = "https://www.crummy.com/software/BeautifulSoup/bs4/doc/"

CHROMA_PERSIST_DIRECTORY = "./chroma_db"
CHROMA_COLLECTION_NAME = "bs4_documentation_search"

EMBEDDING_FUNCTION = OpenAIEmbeddings()
