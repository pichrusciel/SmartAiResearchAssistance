import os
import asyncio
from pathlib import Path
from .researcher import AIResearchAssistant #When you run your project as a package (using uv run), Python expects absolute imports. Need to add dot
from .researcher import print_research_response
from .researcher import ResearchResponse
from .document_loader import load_file_content
from .structlogger import logger 

from dotenv import load_dotenv

import shutil

load_dotenv()


async def main():
    ### Smart AI Research Assistant ###
    logger.info("Smart AI Research Assistant - started")

    persist_directory = "./database/chroma/research_db"
    chunk_size: int = 1000
    chunk_overlap: int = 200

    shutil.rmtree(persist_directory, ignore_errors=True)
    assistant = AIResearchAssistant(persist_directory, chunk_size, chunk_overlap)
    
    # Add research docs from files
    documents_dir = Path("./documents")
    print(f"Documents storage: {documents_dir.absolute()}")
    logger.info("Documents storage", documents_dir=documents_dir.absolute().as_posix())

    if not documents_dir.exists() or not documents_dir.is_dir():
        print(f"Error: Target directory '{documents_dir.absolute()}' does not exist.")
        return

    # Iterate over every item found in the documents directory
    for file_path in documents_dir.iterdir():
        if file_path.is_file():
            # Identify extension, run specific Loader, and return string content
            file_content = load_file_content(file_path)

            if file_content:
                # Execute assistant.add_text using full file name as source parameter    
                assistant.add_text(
                    text=file_content,
                    source=file_path.name
                )

    print(f"\nIndexed: {assistant.get_document_count()} chunks")

    session = "structured_demo"

    # --- Step 1: String vs Structured comparison ---
    print("\n" + "=" * 60)
    print("STEP 1: String response vs Structured response")
    print("=" * 60)

    question = "What is RAG and what are its benefits?"

    print("\n--- String response (ask) ---")
    #string_response = assistant.ask(question,"string_test")
    string_response = await assistant.ask(question, "string_test")
    print(f"Type: {type(string_response)}")
    print(f"Response: {string_response[:200]}...")

    print("\n--- Structured response (ask_structured) ---")
    #structured_response = assistant.ask_structured(question, "struct_test")
    structured_response = await assistant.ask_structured(question, "struct_test")
    print(f"Type: {type(structured_response)}")
    print(f"answer:             {structured_response.answer[:100]}...")
    print(f"confidence:         {structured_response.confidence}")
    print(f"sources:            {structured_response.sources}")
    print(f"key_quotes:         {structured_response.key_quotes[:2]}")
    print(f"follow_up_questions: {structured_response.follow_up_questions}")    

    # --- Step 2: Access fields directly ---
    print("\n" + "=" * 60)
    print("STEP 2: Use fields in code")
    print("=" * 60)

    #r = assistant.ask_structured("What is the attention mechanism?", session)
    r = await assistant.ask_structured("What is the attention mechanism?", session)

    if r.confidence == "high":
        print(f"\n  Confident answer from: {', '.join(r.sources)}")
    else:
        print(f"\n  Low confidence -- may need more sources")

    print(f"\n Answer: {r.answer[:200]}")

    print(f"\n  Suggested follow-ups:")
    for fq in r.follow_up_questions:
        print(f"    -> {fq}")

    # --- Step 3: Multi-turn with structured output ---
    print("\n" + "=" * 60)
    print("STEP 3: Memory works with structured output too")
    print("=" * 60)

    q1 = "What are the components of RAG?"
    print(f"\nUser: {q1}")
    #r1 = assistant.ask_structured(q1, session)
    r1 = await assistant.ask_structured(q1, session)
    print_research_response(q1, r1)    

    q2 = "How does the second component work?"
    print(f"\n{'- '*30}")
    print(f"\nUser: {q2}")
    #r2 = assistant.ask_structured(q2, session)
    r2 = await assistant.ask_structured(q2, session)
    print_research_response(q2, r2)

    q3 = "Connect everything we discussed to LangChain."
    print(f"\n{'- '*30}")
    print(f"\nUser: {q3}")
    #r3 = assistant.ask_structured(q3, session)
    r3 = await assistant.ask_structured(q3, session)
    print_research_response(q3, r3)

    # --- Step 4: Summary stats ---
    print("\n" + "=" * 60)

    history = assistant._get_session_history(session)
    msg_count = len(history.messages) if hasattr(history, "messages") else len(history)

    print(
        f"""
  Document ingestion    -> {assistant.get_document_count()} chunks indexed
  Sources tracked       -> {assistant.list_sources()}
  Basic retrieval       -> similarity search
  Advanced retrieval    -> multi-query + compression
  Conversation memory   -> {msg_count} messages in session '{session}'
  Structured output     -> ResearchResponse with {len(ResearchResponse.model_fields)} fields

  From raw text to a production-ready research assistant.
  That's the full RAG pipeline.
    """
    )

    # Cleanup
    shutil.rmtree(persist_directory, ignore_errors=True)
    logger.info("Smart AI Research Assistant - finished")

def run_app():
    """Synchronous entry point registered in pyproject.toml"""
    asyncio.run(main())

if __name__ == "__main__":
    #main()
    run_app()

