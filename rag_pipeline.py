from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# 1. Configuration & Setup
load_dotenv()
video_id = "Gfr50f6ZBvo" # Replace with your target video ID

def fetch_transcript(video_id: str) -> str:
    """Fetches and joins the transcript for a given YouTube video ID."""
    try:
        print(f"Fetching transcript for video ID: {video_id}")
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
        transcript = " ".join(chunk.text for chunk in transcript_list)
        return transcript
    except TranscriptsDisabled:
        print("Error: No captions available for this video.")
        return ""
    except Exception as e:
        print(f"An error occurred while fetching the transcript: {e}")
        return ""

def format_docs(retrieved_docs):
    """Formats retrieved document chunks into a single text string."""
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

def main():
    # --- Step 1a: Document Ingestion ---
    transcript = fetch_transcript(video_id)
    if not transcript:
        return # Exit if we couldn't get a transcript
    
    print(f"Transcript fetched successfully. Length: {len(transcript)} characters.")

    # --- Step 1b: Text Chunking ---
    print("Chunking transcript...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.create_documents([transcript])
    print(f"Created {len(chunks)} chunks.")

    # --- Step 1c & 1d: Embedding & Vector Storage ---
    print("Generating embeddings and building FAISS vector store...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)

    # --- Step 2: Retrieval Setup ---
    print("Configuring retriever...")
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # --- Step 3: Augmentation & Prompt Setup ---
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    
    prompt = PromptTemplate(
        template="""
          You are a helpful assistant.
          Answer ONLY from the provided transcript context.
          If the context is insufficient, just say you don't know.

          {context}
          Question: {question}
        """,
        input_variables=['context', 'question']
    )

    # --- Step 4: LCEL Chain Composition ---
    print("Assembling LCEL chain...")
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()    
    })
    
    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | llm | parser

    # --- Step 5: Execution ---
    query = 'Can you summarize the video'
    print(f"\nExecuting query: '{query}'...")
    response = main_chain.invoke(query)
    
    print("\n--- Response ---")
    print(response)

if __name__ == "__main__":
    main()
