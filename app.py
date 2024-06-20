from flask import Flask, render_template, request, jsonify
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM,pipeline, AutoTokenizer, BitsAndBytesConfig
# from langchain import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
import torch
import config as ctg

app = Flask(__name__)

def similarity_search(vector_database, text):
    searchDocs = vector_database.similarity_search(text)
    print('Similar Chunk:::')
    print(searchDocs[0].page_content)


def create_rag(dataset_dir, embed_model_path, llm_model_path): 
    encode_kwargs = {'normalize_embeddings':False}
    model_kwargs = {'device':'cuda:0'}
    embeddings = HuggingFaceEmbeddings(
    model_name = embed_model_path,  
    model_kwargs = model_kwargs,
    encode_kwargs=encode_kwargs
    )
    db = FAISS.load_local(dataset_dir, embeddings)
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(llm_model_path,
                                                # load_in_8bit=True,
                                                # quantization_config=bnb_config,
                                                device_map="auto",
                                                torch_dtype=torch.bfloat16,
                                                attn_implementation="flash_attention_2",
                                                max_length = 64)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    llm = HuggingFacePipeline(
        pipeline = pipe,
        model_kwargs={"temperature": ctg.temperature, "max_length": ctg.generation_max_len},
    )
    template = """[INST] <>
あなたは誠実で優秀な日本の弁護士です。
法的な質問に答えるには、マークダウン形式の次のコンテキスト情報と以前のチャット履歴を使用します。 答えがわからない場合は、答えをでっち上げようとせず、ただわからないと言ってください。 回答はできるだけ簡潔にしてください。
<>

チャット履歴 : {history}

コンテキスト : {context}

質問: {question}
[/INST]"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["history", "context", "question"],template=template)
    memory = ConversationBufferMemory(memory_key='history', input_key="question")

    chain = RetrievalQA.from_chain_type(   
    llm=llm,   
    chain_type="stuff",   
    retriever=db.as_retriever(),   
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT, 'memory':memory} 
    )
    return chain

def generate_reply(chain, user_input):
    # Implement your logic to generate a bot reply based on user_input
    # This is where you call your Python function
    # For now, let's just echo the user's input
    # return f"Bot says: I received '{user_input}'"
    response = str(chain.run(user_input))
    return response[response.rfind('[/INST]')+7:]

@app.route('/')
def index(): 
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    user_input = request.json.get('user_input', '')
    
    # Implement your logic to process user_input and generate a bot reply
    bot_reply = generate_reply(chain=chain, user_input=user_input)

    return jsonify({'bot_reply': bot_reply})



if __name__ == '__main__':
    chain = create_rag(dataset_dir=ctg.dataset_dir, embed_model_path=ctg.embed_model_path, llm_model_path=ctg.llm_model_path)
    app.run(debug=True)
