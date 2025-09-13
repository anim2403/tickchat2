import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from groq import Groq
from pinecone import Pinecone




ALLOWED_RAG_TOPICS = {
    "How-to",
    "Product",
    "Best practices",
    "API/SDK",
    "SSO",
}
_ALLOWED_NORM = {t.lower() for t in ALLOWED_RAG_TOPICS}


def _norm(tag: str) -> str:
    return (tag or "").strip().lower()


class AtlanRAGPipeline:
    def debug_index_status(self):
        
        try:
            stats = self.index.describe_index_stats()
            print(" Index Stats:")
            print(f"  - Total vectors: {stats.get('total_vector_count', 0)}")
            print(f"  - Index fullness: {stats.get('index_fullness', 0)}")
            print(f"  - Dimension: {stats.get('dimension', 0)}")

            index_dimension = stats.get('dimension', 384)
            model_dimension = self.embedder.get_sentence_embedding_dimension()

            print(" Embedding Check:")
            print(f"  - Model dimension: {model_dimension}")
            print(f"  - Index dimension: {index_dimension}")

            if model_dimension != index_dimension:
                print(f" WARNING: Dimension mismatch! Model: {model_dimension}, Index: {index_dimension}")
                print("   This could cause poor search results. Consider using a model with the index's dimension.")

            
            test_vector = [0.1] * index_dimension  
            test_results = self.index.query(
                vector=test_vector,
                top_k=3,
                include_metadata=True,
                namespace="atlan",
            )

            print(f" Test query returned {len(test_results.matches)} results")
            for i, match in enumerate(test_results.matches[:2]):
                print(f"  Sample {i+1}:")
                print(f"    Score: {match.score}")
                md = match.metadata or {}
                print(f"    Metadata keys: {list(md.keys()) if md else 'None'}")
                if md.get('text'):
                    text_preview = md['text'][:100].replace('\n', ' ')
                    print(f"    Text preview: {text_preview}...")
                if md.get('url'):
                    print(f"    URL: {md['url']}")

        except Exception as e:
            print(f" Error checking index status: {e}")
            import traceback
            print(f" Full error trace: {traceback.format_exc()}")

    def __init__(self, pinecone_api_key: str = None, groq_api_key: str = None,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the RAG pipeline with Pinecone vector database.

        Args:
            pinecone_api_key: Pinecone API key (will try to get from env if not provided)
            groq_api_key: Groq API key (will try to get from env if not provided)
            embedding_model: The sentence transformer model to use for embeddings
        """
        
        self.pc = Pinecone(api_key=pinecone_api_key or os.getenv("PINECONE_API_KEY"))
        self.index_name = "atlan-docs"

        
        self.groq_client = Groq(api_key=groq_api_key or os.getenv("GROQ_API_KEY"))

        
        self.embedding_model = embedding_model
        self.embedder = SentenceTransformer(embedding_model)

        
        try:
            self.index = self.pc.Index(self.index_name)
            print(f" Connected to Pinecone index: {self.index_name}")

            stats = self.index.describe_index_stats()
            print(f" RAG Pipeline initialized with {stats.get('total_vector_count', 0)} documents")

            self.debug_index_status()

        except Exception as e:
            print(f" Error connecting to Pinecone index '{self.index_name}': {e}")
            print("Make sure the index exists and your API key is correct.")
            raise

   
    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents based on the query using Pinecone.

        Args:
            query: User query string
            top_k: Number of documents to retrieve

        Returns:
            List of relevant documents with scores and metadata
        """
        try:
            
            query_embedding = self.embedder.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True
            )[0].tolist()

            
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace="atlan",
                include_metadata=True,
                include_values=False
            )

            
            results = []
            for match in search_results.matches:
                md = match.metadata or {}
                results.append({
                    "score": float(match.score),
                    "text": md.get("text", ""),
                    "url": md.get("url", ""),
                    "chunk": md.get("chunk", "")
                })

            return results

        except Exception as e:
            print(f" Error retrieving documents: {e}")
            return []

   
    def should_use_rag(self, topic_tags: List[str]) -> bool:
        """
        Use RAG if ANY of the topic tags is in the allowed set.
        If the list is empty or contains only disallowed topics, do NOT use RAG.
        """
        tags = topic_tags or []
        return any(_norm(t) in _ALLOWED_NORM for t in tags)

    
    def generate_answer(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Generate an answer using Groq LLM with retrieved context.

        Args:
            query: Original user query/ticket
            context_docs: Retrieved relevant documents

        Returns:
            Generated answer
        """
        # Prepare context from retrieved documents
        context_texts = []
        sources = []

        for i, doc in enumerate(context_docs):
            text = (doc or {}).get("text")
            if text:
                context_texts.append(f"Document {i+1}:\n{text}")
                url = doc.get("url")
                if url:
                    sources.append(f"- [{url}]({url})")

        if not context_texts:
            return "No relevant documentation found to answer your question."

        context = "\n\n".join(context_texts)
        sources_text = "\n".join(sources) if sources else "No sources available"

        system_prompt = """You are an expert Atlan support agent helping users with their questions about Atlan's data catalog platform.

Use the provided documentation context to answer the user's question accurately and helpfully. Follow these guidelines:

1. Provide clear, step-by-step instructions when applicable
2. Include relevant code examples if available in the context
3. Reference specific Atlan features and capabilities
4. If the context doesn't fully answer the question, acknowledge this limitation
5. Keep your response focused and practical
6. Format your response with proper headings and bullet points for readability

Always base your answer primarily on the provided context from Atlan's official documentation."""

        user_prompt = f"""Question/Ticket: {query}

Context from Atlan Documentation:
{context}

Please provide a comprehensive answer based on the documentation context above."""

        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )

            answer = response.choices[0].message.content
            return f"{answer}\n\n**Sources:**\n{sources_text}"

        except Exception as e:
            return f"Error generating answer: {str(e)}"

    
    def get_ai_help(self, ticket_subject: str, ticket_body: str, topic_tags: List[str]) -> Dict[str, Any]:
        """
        Main method to get AI help for a ticket.

        Policy:
        - If ANY allowed topic is present -> use RAG and return a direct answer.
        - If none of the allowed topics are present (empty or only disallowed) -> route with a simple message.
        """
        topic_tags = topic_tags or []
        has_allowed = self.should_use_rag(topic_tags)

        if not has_allowed:
            routed_as = topic_tags[0] if topic_tags else "General"
            return {
                "use_rag": False,
                "message": f"This ticket has been classified as a '{routed_as}' issue and routed to the appropriate team.",
                "topic_tags": topic_tags,
                "allowed_topics": sorted(ALLOWED_RAG_TOPICS),
            }

        
        full_query = f"Subject: {ticket_subject}\n\nDescription: {ticket_body}"

        
        relevant_docs = self.retrieve_documents(full_query, top_k=5)
        if not relevant_docs:
            return {
                "use_rag": True,
                "error": "No relevant documentation found for this query.",
                "topic_tags": topic_tags,
                "relevant_docs": [],
                "num_sources": 0,
            }

        
        answer = self.generate_answer(full_query, relevant_docs)
        return {
            "use_rag": True,
            "answer": answer,
            "relevant_docs": relevant_docs,
            "topic_tags": topic_tags,
            "num_sources": len(relevant_docs),
        }



def test_rag_pipeline():
    """Test the RAG pipeline with some sample queries"""
    try:
        rag = AtlanRAGPipeline()

        test_cases = [
            {
                "subject": "How to create a new data asset",
                "body": "I need help creating a new data asset in Atlan. What are the steps?",
                "topic_tags": ["How-to", "Product"],  
            },
            {
                "subject": "API authentication issue",
                "body": "I'm having trouble authenticating with the Atlan API. Can you help?",
                "topic_tags": ["API/SDK"],  
            },
            {
                "subject": "Connector not working",
                "body": "My Snowflake connector is not syncing properly",
                "topic_tags": ["Connector", "Technical"],  
            },
            {
                "subject": "Missing tags should route",
                "body": "Just checking behavior when tags are empty.",
                "topic_tags": [],  
            },
        ]

        for i, test_case in enumerate(test_cases):
            print(f"\n--- Test Case {i+1} ---")
            print(f"Subject: {test_case['subject']}")
            print(f"Topic Tags: {test_case['topic_tags']}")

            result = rag.get_ai_help(
                test_case["subject"],
                test_case["body"],
                test_case["topic_tags"]
            )

            if result.get("use_rag"):
                if "answer" in result:
                    print(f"AI Answer: {result['answer'][:200]}...")
                    print(f"Number of sources used: {result['num_sources']}")
                else:
                    print(f"Error: {result.get('error', 'Unknown error')}")
            else:
                print(f"Message: {result['message']}")

    except Exception as e:
        print(f" Test failed: {e}")


if __name__ == "__main__":
    test_rag_pipeline()
