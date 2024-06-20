dir_path = '/media/nsl3090-3/hdd1/hujaifa/JP_NSL_Lawyer/Legal Text'
dataset_dir = '/media/nsl3090-3/hdd1/hujaifa/Langchain/vectorstore/db_jp_law_512'

vector_chunk_size=512
vector_overlap = 20
vector_separator = '\n\n\n'

embed_model_path = 'intfloat/multilingual-e5-large'
llm_model_path = 'elyza/ELYZA-japanese-Llama-2-7b-fast-instruct'
generation_max_len = 512
temperature = 0