import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

import json

# Configure Google Gemini API
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])


# Original PDF processing functions
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def process_user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )

        st.write("Reply: ", response["output_text"])
    except FileNotFoundError:
        st.error("Please process some documents first before asking questions.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


# New MCQ Generation functions
def generate_mcqs(text, num_questions=5):
    try:
        model = ChatGoogleGenerativeAI(model="gemini-pro",
                                       temperature=0.3)  # Reduced temperature for more consistent output

        prompt = f"""
        You are a precise MCQ generator that creates exactly {num_questions} multiple choice questions. 
        Follow this exact JSON structure and formatting rules:

        1. Start the response with a JSON array
        2. Each question must have these exact fields:
           - "question_text": A clear question as a string
           - "choices": An array of exactly 4 string choices
           - "correct_answer": A number from 0 to 3 indicating the correct choice
           - "explanation": A string explaining why the correct answer is right

        Example of EXACT format required:
        [
            {{
                "question_text": "What is the capital of France?",
                "choices": ["London", "Paris", "Berlin", "Madrid"],
                "correct_answer": 1,
                "explanation": "Paris is the capital city of France."
            }},
            {{
                "question_text": "Which planet is closest to the Sun?",
                "choices": ["Venus", "Mars", "Mercury", "Earth"],
                "correct_answer": 2,
                "explanation": "Mercury is the closest planet to the Sun in our solar system."
            }}
        ]

        Generate {num_questions} questions in this EXACT format based on this text: {text[:5000]}

        Important:
        - Use double quotes for all strings
        - Include exactly 4 choices for each question
        - Make sure correct_answer is a number (0-3)
        - Do not include any explanatory text before or after the JSON
        - Make questions relevant to the key concepts in the provided text
        """

        response = model.invoke(prompt)

        # Clean the response content
        content = response.content.strip()
        content = content.replace('```json', '').replace('```', '').strip()

        try:
            mcqs = json.loads(content)

            # Detailed validation with specific error messages
            if not isinstance(mcqs, list):
                raise ValueError("Response is not a list of questions")

            for i, mcq in enumerate(mcqs):
                # Check all required fields exist
                required_fields = ['question_text', 'choices', 'correct_answer', 'explanation']
                missing_fields = [field for field in required_fields if field not in mcq]
                if missing_fields:
                    raise ValueError(f"Question {i + 1} is missing fields: {', '.join(missing_fields)}")

                # Validate field types and values
                if not isinstance(mcq['question_text'], str):
                    raise ValueError(f"Question {i + 1}: question_text must be a string")

                if not isinstance(mcq['choices'], list):
                    raise ValueError(f"Question {i + 1}: choices must be a list")

                if len(mcq['choices']) != 4:
                    raise ValueError(f"Question {i + 1}: must have exactly 4 choices")

                if not all(isinstance(choice, str) for choice in mcq['choices']):
                    raise ValueError(f"Question {i + 1}: all choices must be strings")

                if not isinstance(mcq['correct_answer'], int):
                    raise ValueError(f"Question {i + 1}: correct_answer must be an integer")

                if not 0 <= mcq['correct_answer'] <= 3:
                    raise ValueError(f"Question {i + 1}: correct_answer must be between 0 and 3")

                if not isinstance(mcq['explanation'], str):
                    raise ValueError(f"Question {i + 1}: explanation must be a string")

            return mcqs

        except json.JSONDecodeError as e:
            st.error(f"Failed to parse JSON response from model")
            st.expander("Debug Information").code(content, language="json")
            return None

    except Exception as e:
        st.error(f"Error generating MCQs: {str(e)}")
        if 'content' in locals():
            st.expander("Model Response Debug").code(content)
        return None


def log_mcq_debug(mcqs, response_content):
    """Helper function to log debugging information"""
    st.expander("MCQ Generation Debug Info").write({
        "Number of questions": len(mcqs) if mcqs else 0,
        "Raw response": response_content,
        "Parsed MCQs": mcqs
    })


def create_quiz_interface(mcqs):
    if not mcqs:
        return

    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = [-1] * len(mcqs)

    st.subheader("📝 Quiz Time!")

    for i, mcq in enumerate(mcqs):
        st.write(f"\n**Question {i + 1}:** {mcq['question_text']}")

        choice = st.radio(
            f"Select your answer for question {i + 1}:",
            mcq['choices'],
            key=f"q_{i}",
            index=st.session_state.user_answers[i] if st.session_state.user_answers[i] != -1 else None
        )

        if choice:
            st.session_state.user_answers[i] = mcq['choices'].index(choice)


def show_results(mcqs):
    if not mcqs or 'user_answers' not in st.session_state:
        return

    correct_count = 0
    st.subheader("📊 Quiz Results")

    for i, (mcq, user_ans) in enumerate(zip(mcqs, st.session_state.user_answers)):
        is_correct = user_ans == mcq['correct_answer']
        if is_correct:
            correct_count += 1

        st.write(f"\n**Question {i + 1}:** {mcq['question_text']}")
        st.write(f"Your answer: {mcq['choices'][user_ans]}")
        st.write(f"Correct answer: {mcq['choices'][mcq['correct_answer']]}")

        if is_correct:
            st.success("✅ Correct!")
        else:
            st.error("❌ Incorrect")

        st.info(f"Explanation: {mcq['explanation']}")

    score_percentage = (correct_count / len(mcqs)) * 100
    st.subheader("🏆 Final Score")
    st.write(f"You got {correct_count} out of {len(mcqs)} questions correct ({score_percentage:.1f}%)")


def main():
    st.set_page_config(page_title="Document Chat & Quiz", page_icon="📚", layout="wide")
    st.header("Document Chat & Quiz Generator 📝")

    # Create tabs
    tab1, tab2 = st.tabs(["💬 Chat with PDF", "📝 Generate Quiz"])

    # Tab 1: Original PDF Chat functionality
    with tab1:
        with st.sidebar:
            st.title("📄 Document Upload")
            pdf_docs = st.file_uploader(
                "Upload PDF Files and Click on Process",
                accept_multiple_files=True,
                type=['pdf']
            )

            if st.button("Process PDFs", type="primary"):
                if not pdf_docs:
                    st.error("Please upload at least one PDF file.")
                    return

                with st.spinner("Processing Documents..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.session_state.current_text = raw_text
                        st.success("✅ Documents processed successfully!")
                        st.session_state.docs_processed = True
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
                        st.session_state.docs_processed = False

        # Chat interface
        user_question = st.text_input(
            "Ask a question about your documents:",
            placeholder="What would you like to know about the uploaded PDFs?"
        )

        if user_question:
            if not 'docs_processed' in st.session_state or not st.session_state.docs_processed:
                st.warning("Please process some documents first before asking questions.")
                return
            process_user_input(user_question)

    # Tab 2: Quiz Generation
    with tab2:
        if 'docs_processed' in st.session_state and st.session_state.docs_processed:
            if 'mcqs' not in st.session_state:
                if st.button("Generate Quiz"):
                    with st.spinner("Generating quiz questions..."):
                        st.session_state.mcqs = generate_mcqs(st.session_state.current_text)
                        if st.session_state.mcqs:
                            st.success("Quiz generated successfully!")

            if 'mcqs' in st.session_state and st.session_state.mcqs:
                create_quiz_interface(st.session_state.mcqs)

                if st.button("Submit Quiz"):
                    if -1 in st.session_state.user_answers:
                        st.warning("Please answer all questions before submitting.")
                    else:
                        show_results(st.session_state.mcqs)

                if st.button("New Quiz"):
                    st.session_state.pop('mcqs', None)
                    st.session_state.pop('user_answers', None)
                    st.rerun()
        else:
            st.warning("Please process some PDF documents first to generate a quiz.")


if __name__ == "__main__":
    main()