import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import json

# Configure Google Gemini API
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])


# PDF Processing Functions
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


# Enhanced MCQ Generation Functions
def generate_mcqs(text, num_questions=5, difficulty="medium"):
    try:
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

        difficulty_specs = {
            "easy": "Focus on basic concepts and direct facts. Questions should be straightforward with exactly one correct answer and three clearly incorrect options.",
            "medium": "Include application of concepts and analytical thinking. Each question must have exactly one correct answer and three plausible but definitely incorrect options.",
            "hard": "Create complex questions involving analysis and synthesis of concepts. Maintain exactly one correct answer with three challenging but definitively incorrect distractors."
        }

        prompt = f"""
        Generate exactly {num_questions} multiple choice questions at {difficulty} difficulty level.

        Critical Rules for Answer Choices:
        1. Each question MUST have EXACTLY ONE correct answer
        2. The other three options MUST be definitively incorrect
        3. Avoid partially correct answers or "all of the above" options
        4. Make incorrect options plausible but clearly wrong
        5. Avoid overlapping or ambiguous answer choices
        6. Ensure options are mutually exclusive

        Difficulty Guidelines:
        {difficulty_specs[difficulty]}

        Follow this exact JSON structure:
        [
            {{
                "question_text": "What is the capital of France?",
                "choices": [
                    "London (incorrect)",
                    "Paris (correct)",
                    "Berlin (incorrect)",
                    "Madrid (incorrect)"
                ],
                "correct_answer": 1,
                "explanation": "Paris is the capital city of France. While the other options are European capitals, they are capitals of different countries: London (UK), Berlin (Germany), and Madrid (Spain).",
                "difficulty": "{difficulty}"
            }}
        ]

        Important Rules:
        1. Questions must match the specified {difficulty} difficulty level
        2. Each question must test a different concept
        3. All options must be complete sentences or phrases
        4. Avoid using "none of the above" or "all of the above"
        5. Make each incorrect option independently wrong
        6. Explanations must clearly state why the correct answer is right and why others are wrong

        Generate {num_questions} questions based on this text: {text[:5000]}
        """

        response = model.invoke(prompt)
        content = response.content.strip()
        content = content.replace('```json', '').replace('```', '').strip()

        try:
            mcqs = json.loads(content)

            # Validation
            if not isinstance(mcqs, list):
                raise ValueError("Response is not a list of questions")

            for i, mcq in enumerate(mcqs):
                required_fields = ['question_text', 'choices', 'correct_answer', 'explanation', 'difficulty']
                missing_fields = [field for field in required_fields if field not in mcq]

                if missing_fields:
                    raise ValueError(f"Question {i + 1} is missing fields: {', '.join(missing_fields)}")

                if not isinstance(mcq['question_text'], str):
                    raise ValueError(f"Question {i + 1}: question_text must be a string")

                if not isinstance(mcq['choices'], list) or len(mcq['choices']) != 4:
                    raise ValueError(f"Question {i + 1}: must have exactly 4 choices")

                if not isinstance(mcq['correct_answer'], int) or not 0 <= mcq['correct_answer'] <= 3:
                    raise ValueError(f"Question {i + 1}: correct_answer must be between 0 and 3")

                if mcq['difficulty'] != difficulty:
                    raise ValueError(f"Question {i + 1}: incorrect difficulty level")

            return mcqs

        except json.JSONDecodeError as e:
            st.error("Failed to parse JSON response")
            st.expander("Debug Information").code(content, language="json")
            return None

    except Exception as e:
        st.error(f"Error generating MCQs: {str(e)}")
        if 'content' in locals():
            st.expander("Model Response Debug").code(content)
        return None


def create_quiz_interface(mcqs):
    if not mcqs:
        return

    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = [-1] * len(mcqs)

    st.subheader("📝 Quiz Time!")

    if mcqs and 'difficulty' in mcqs[0]:
        st.info(f"Difficulty Level: {mcqs[0]['difficulty'].upper()}")

    for i, mcq in enumerate(mcqs):
        st.write(f"\n**Question {i + 1} of {len(mcqs)}:** {mcq['question_text']}")

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

    # Tab 1: Chat with PDF
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
                        st.session_state.docs_processed = True
                        st.success("✅ Documents processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
                        st.session_state.docs_processed = False

        user_question = st.text_input(
            "Ask a question about your documents:",
            placeholder="What would you like to know about the uploaded PDFs?"
        )

        if user_question:
            if not hasattr(st.session_state, 'docs_processed') or not st.session_state.docs_processed:
                st.warning("Please process some documents first before asking questions.")
                return
            process_user_input(user_question)

    # Tab 2: Quiz Generation
    with tab2:
        if hasattr(st.session_state, 'docs_processed') and st.session_state.docs_processed:
            col1, col2 = st.columns(2)

            with col1:
                num_questions = st.slider("Number of Questions", min_value=5, max_value=10, value=5)

            with col2:
                difficulty = st.select_slider(
                    "Select Difficulty Level",
                    options=["easy", "medium", "hard"],
                    value="medium"
                )

            if 'mcqs' not in st.session_state:
                if st.button("Generate Quiz"):
                    with st.spinner("Generating quiz questions..."):
                        st.session_state.mcqs = generate_mcqs(
                            st.session_state.current_text,
                            num_questions=num_questions,
                            difficulty=difficulty
                        )
                        if st.session_state.mcqs:
                            st.success("Quiz generated successfully!")

            if 'mcqs' in st.session_state and st.session_state.mcqs:
                create_quiz_interface(st.session_state.mcqs)

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Submit Quiz"):
                        if -1 in st.session_state.user_answers:
                            st.warning("Please answer all questions before submitting.")
                        else:
                            show_results(st.session_state.mcqs)

                with col2:
                    if st.button("New Quiz"):
                        st.session_state.pop('mcqs', None)
                        st.session_state.pop('user_answers', None)
                        st.rerun()
        else:
            st.warning("Please process some PDF documents first to generate a quiz.")


if __name__ == "__main__":
    main()