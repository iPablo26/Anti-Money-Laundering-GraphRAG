import logging
import os
from typing import List

from google.cloud import bigquery
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration (supports environment variable overrides)
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "build-with-ai-488112")
REGION = os.getenv("GCP_REGION", "us-central1")
BQ_DATASET_ID = os.getenv("BQ_DATASET_ID", "fingraph_rag")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Initialize BigQuery client
bq_client = bigquery.Client(project=GCP_PROJECT_ID)


class FinGraphRetriever(BaseRetriever):
    """Hybrid GraphRAG retriever combining BigQuery vector search and GQL traversal."""

    project: str
    dataset: str

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        # 1. Semantic Vector Search using Cosine Distance
        logger.info("Step 1/3: Running semantic vector search over AccountAudits...")
        vector_query = f"""
            SELECT id, audit_details
            FROM `{self.project}.{self.dataset}.AccountAudits`
            ORDER BY COSINE_DISTANCE(
                embedding,
                (
                    SELECT embedding
                    FROM AI.GENERATE_EMBEDDING(
                        MODEL `{self.project}.{self.dataset}.embedding_model`,
                        (SELECT @query AS content)
                    )
                )
            )
            LIMIT 1
        """
        res = bq_client.query(
            vector_query,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("query", "STRING", query)
                ]
            ),
        ).result()

        start_id = None
        audit_text = ""
        for row in res:
            start_id = row.id
            audit_text = row.audit_details

        if start_id is None:
            logger.warning("No matching account audit entry found for query.")
            return []

        logger.info("Matched target Account ID=%s via vector search.", start_id)

        # 2. Native Graph Traversal (BigQuery GQL MATCH)
        logger.info("Step 2/3: Traversing FinGraph relationships for Account ID=%s...", start_id)
        graph_query = f"""
            GRAPH `{self.project}.{self.dataset}.FinGraph`
            MATCH
              (sender_person:Person)-[:Owns]->(sender_acc:Account)
              -[tx:Transfers]->
              (a:Account)
              -[repays:Repays]->(l:Loan),
              (owner:Person)-[:Owns]->(a)
            WHERE a.id = @id
            RETURN
              owner.name AS owner_name,
              a.type AS account_type,
              sender_person.name AS sender_name,
              tx.amount AS transfer_amount,
              repays.amount AS repayment_amount,
              l.id AS loan_id
        """
        graph_res = bq_client.query(
            graph_query,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("id", "INT64", start_id)
                ]
            ),
        ).result()

        context_docs = [
            Document(page_content=f"Primary Audit Log (Target Account): {audit_text}")
        ]
        sender_names = []
        for row in graph_res:
            sender_names.append(row["sender_name"])
            doc_str = (
                f"Account Owner: {row['owner_name']} (Account Type: {row['account_type']}). "
                f"Received transfer of ${row['transfer_amount']} from {row['sender_name']}. "
                f"Made loan repayment of ${row['repayment_amount']} to Loan {row['loan_id']}."
            )
            context_docs.append(Document(page_content=doc_str))

        # 3. Upstream Counterparty Audit Enrichment
        if sender_names:
            unique_senders = list(dict.fromkeys(sender_names))
            logger.info(
                "Step 3/3: Fetching audit logs for %d upstream sender(s)...",
                len(unique_senders),
            )
            sender_audit_query = f"""
                SELECT p.name, au.audit_details
                FROM `{self.project}.{self.dataset}.AccountAudits` au
                JOIN `{self.project}.{self.dataset}.Account` a ON au.id = a.id
                JOIN `{self.project}.{self.dataset}.PersonOwnAccount` poa ON a.id = poa.account_id
                JOIN `{self.project}.{self.dataset}.Person` p ON poa.id = p.id
                WHERE p.name IN UNNEST(@sender_names)
            """
            sender_audits = bq_client.query(
                sender_audit_query,
                job_config=bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ArrayQueryParameter(
                            "sender_names", "STRING", unique_senders
                        )
                    ]
                ),
            ).result()
            for row in sender_audits:
                context_docs.append(
                    Document(
                        page_content=f"Audit Log for Sender {row['name']}: {row['audit_details']}"
                    )
                )

        return context_docs


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# Initialize LLM (Vertex AI backend via unified google-genai SDK) & Retriever
llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    project=GCP_PROJECT_ID,
    location=REGION,
    vertexai=True,
)
retriever = FinGraphRetriever(project=GCP_PROJECT_ID, dataset=BQ_DATASET_ID)

prompt = ChatPromptTemplate.from_template("""
You are a Lead Fraud Analyst. Use the following audit logs and graph transaction history to answer the question.
Your goal is to connect the dots between the entities and explain the flow of funds.
If you see transfers from flagged users or shell companies, highlight the money laundering risk.

Context:
{context}

Question: {question}

Detailed Fraud Report:
""")

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    question = "Why is Menville's loan repayment at risk? Flag any suspicious activity if you notice."
    logger.info("Investigating question: %s", question)
    print(chain.invoke(question))
