from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import config as ctg


if __name__ == "__main__":
    encode_kwargs = {'normalize_embeddings':False}
    model_kwargs = {'device':'cuda:0'}
    embeddings = HuggingFaceEmbeddings(
    model_name = ctg.embed_model_path,  
    model_kwargs = model_kwargs,
    encode_kwargs=encode_kwargs
    )
    dirLoader = DirectoryLoader(ctg.dir_path, glob='**/*.txt', loader_cls=TextLoader)#, use_multithreading=True)
    documents = dirLoader.load()
    print('Number of Documents:::')
    print(len(documents))

    # text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=ctg.vector_chunk_size, chunk_overlap=ctg.vector_overlap, separators=ctg.vector_separator)
    docs = text_splitter.split_documents(documents)


    db = None
    for doc in tqdm(docs):
        if db:
            db.add_documents([doc])
        else:
            db = FAISS.from_documents([doc], embeddings)
    db.save_local(ctg.dir_path)