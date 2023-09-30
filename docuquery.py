import streamlit as st
from docx import Document

from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain


class WordSuccessChecker:
    def __init__(self):
        self.uploaded_file = None
        self.doc_text = None
        self.input_text = None

    def query_doc(self):
        if len(self.doc_text.split()) < 5000:
            return self.query_doc_short()
        else:
            return self.query_doc_long()


    def query_doc_short(self):
        _validation_prompt = """
        You are a highly trained assistant who reads through input documents and answer questions about them.
        This is the input document: {doc_text}

        You get the following question: {input_text}

        Provide a accurate and informative answer only based on the document. If question cannot be answered based on the input document do not make up an answer just tell answer is not in the text. """

        VALIDATION_PROMPT = PromptTemplate(
            input_variables=["doc_text", "input_text"],
            template=_validation_prompt
        )

        llm_chain = LLMChain(
            llm=OpenAI(temperature=0, max_tokens=-1, model_name="gpt-3.5-turbo-16k"),
            prompt=VALIDATION_PROMPT
        )

        return llm_chain({"doc_text": self.doc_text, "input_text": self.input_text})["text"]
    
    def query_doc_long(self):
        _validation_prompt = """
        You are a highly trained assistant who reads through input documents and answer questions about them.
        This is the input document: {doc_text}

        You get the following question: {input_text}

        Provide a accurate and informative answer only based on the document. If question cannot be answered based on the input document do not make up an answer just tell answer is not in the text. """

        VALIDATION_PROMPT = PromptTemplate(
            input_variables=["doc_text", "input_text"],
            template=_validation_prompt
        )

        llm_chain = LLMChain(
            llm=OpenAI(temperature=0, max_tokens=-1, model_name="gpt-3.5-turbo-16k"),
            prompt=VALIDATION_PROMPT
        )

        return llm_chain({"doc_text": self.doc_text[:14000], "input_text": self.input_text})["text"]

    def run(self):
        st.image("./logo1.png", width=150)
        st.title("AI Document Assistant")

        # Upload Word document
        self.uploaded_file = st.file_uploader("Upload a Word document", type=["docx"])

        if self.uploaded_file is not None:
            # Read the uploaded Word document
            doc = Document(self.uploaded_file)
            self.doc_text = "\n".join([para.text for para in doc.paragraphs])

            # User input text
            self.input_text = st.text_input("Enter a question to query the uploaded document:")

            if st.button("Query"):
                if self.input_text:
                    result = self.query_doc()
                    st.write(f"Result: {result}")
                else:
                    st.write("Please enter a question.")

if __name__ == '__main__':
    app = WordSuccessChecker()
    app.run()
