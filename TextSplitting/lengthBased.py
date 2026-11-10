from langchain_text_splitters import CharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader

# loader = PyPDFLoader('dl-curriculum.pdf')

# docs = loader.load()


text='''Artificial Intelligence (AI) has rapidly transformed from a futuristic concept into a foundational technology shaping nearly every aspect of modern life. At its core, AI refers to the simulation of human intelligence in machines programmed to think, learn, adapt, and solve complex problems. By leveraging massive datasets and advanced algorithms, AI systems can recognize patterns, generate human-like content, and make autonomous decisions with unprecedented speed and accuracy. From healthcare diagnostics and scientific research to everyday productivity tools and automation, AI empowers society to work smarter and innovate faster. However, as these technologies continue to evolve, they also raise important ethical considerations surrounding privacy, workforce shifts, and responsible use, making a balanced approach to its future development more crucial than ever.'''
splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=''
)

result = splitter.split_text(text)
# result = splitter.split_documents(docs)

# print(result[1].page_content)
print(result)