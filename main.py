import fitz
import ollama
import os
import pandas as pd
from tqdm import tqdm

input_dir = 'input'
questions_filename = 'questions.txt'
output_file='output.csv'

lst_key_query = [
    'What is the person name?',
]

def pdf_to_str(pdf_filename):
    with fitz.open(pdf_filename) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text    

def get_document_fact(text, query):
    if text is None or text == '':
        return ''
    lst_messages = []

    lst_messages.append({'role': 'user', 'content': 'Next message will contain a CV of a person. \
    You will need to answer a question about this CV that will follow.'})
    lst_messages.append({'role': 'user', 'content': text})


    def add_question(str_query):
        lst_messages.append({'role': 'user', 'content': f"""Given the CV provided above, answer the following question: {str_query} \
        Respond with the complete but shortest possible and direct answer as if you were API. Remove meta words from the answer. \
        Don't say things like "Person phone number is ..." or "Person name is ...". or "The name provided in the CV ..."\
        Just provide the value itself."""})

    add_question(query)

    lst_messages.append(ollama.chat(
        model='mistral',
        messages=lst_messages,
        stream=False,
    )['message'])

    return lst_messages[-1]['content'].strip()

def get_document_facts(text, lst_key_query, lst_fact_query, str_colname_id='id'):
    dict_doc_key = {
        k: get_document_fact(text, k) for k in lst_key_query
    }

    dict_doc_ret = {}
    dict_doc_ret[str_colname_id] = ', '.join(dict_doc_key.values())
    
    dict_doc_facts = {
        k: get_document_fact(text, k) for k in lst_fact_query
    }
    
    dict_doc_ret.update(dict_doc_facts)
    
    return dict_doc_ret

def get_pdf_facts(pdf_filename, lst_key_query, lst_fact_query, str_colname_id='id'):
    text = pdf_to_str(pdf_filename)
    dict_facts = get_document_facts(text[:6000], lst_key_query, lst_fact_query, str_colname_id)
    return dict_facts


def process_documents(input_dir, lst_key_query, 
                      questions_filename='questions.txt', 
                      output_file='output.csv'):

    lst_fact_query = [
    ]

    if os.path.exists(questions_filename):
        with open(questions_filename, 'r') as f:
            lst_fact_query = f.read().splitlines()

    lst_doc_filename = os.listdir(input_dir)
    lst_dict_res = []
    for filename in tqdm(lst_doc_filename):
        dict_res = {}
        dict_res['filename'] = filename

        dict_facts = get_pdf_facts(f"{input_dir}/{filename}", 
                                lst_key_query, 
                                lst_fact_query,
                                str_colname_id = 'name'
                                )
        dict_res.update(dict_facts)
        lst_dict_res.append(dict_res)

    df_res = pd.DataFrame(lst_dict_res)
    df_res.to_csv(output_file, index=False)
    return df_res


if __name__ == "__main__":
    process_documents(input_dir, 
                      lst_key_query,
                      questions_filename,
                      output_file
                      )
    
