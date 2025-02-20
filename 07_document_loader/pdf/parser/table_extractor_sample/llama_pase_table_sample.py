from dotenv import load_dotenv
from llama_index.core.node_parser import SimpleNodeParser
from llama_parse import LlamaParse

load_dotenv()

node_parser = SimpleNodeParser()

# file_path = "../../sample2/BM202404290000031873_0.pdf"
file_path = "../../sample2/BM202404110000030261_0.pdf"

documents = LlamaParse(result_type="markdown", disable_ocr=True).load_data(file_path)
print(documents)
