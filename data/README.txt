================================================================================
CORTEX PROJECT — KNOWLEDGE BASE DATASET
Meridian Technologies (Fictional Company)
================================================================================

This dataset contains simulated enterprise policy documents for Meridian
Technologies — a fictional company created for educational purposes. Use these
documents as the knowledge base for your Cortex AI assistant project.

--------------------------------------------------------------------------------
INGESTION
--------------------------------------------------------------------------------

Ingest all documents into your vector database with:

    python rag/ingest.py --data-dir ./data/

Each document includes an "Access Level" field in its header metadata. Your
ingest script should parse this field and store it alongside each document chunk
in your vector database so your RBAC filter can apply it at query time.

--------------------------------------------------------------------------------
ACCESS LEVELS
--------------------------------------------------------------------------------

Meridian documents use four access levels. Your RBAC logic should enforce:

  public      All user tiers can retrieve these documents.
              (anonymous, standard, manager, exec)

  general     Standard employees and above can retrieve these.
              (standard, manager, exec)

  manager     People managers and above only.
              (manager, exec)

  executive   Executive leadership only.
              (exec)

When a user's tier does not match the required access level, the document chunk
should be excluded from retrieval — do not return it in context, even if it is
the highest-similarity match.

--------------------------------------------------------------------------------
DOCUMENT LIST
--------------------------------------------------------------------------------

hr_policies/
  parental_leave_policy.txt          Access Level: public
  performance_review_process.txt     Access Level: public
  remote_work_policy.txt             Access Level: public
  compensation_and_benefits.txt      Access Level: general
  code_of_conduct.txt                Access Level: public

it_policies/
  acceptable_use_policy.txt          Access Level: public
  software_request_process.txt       Access Level: public
  data_security_guidelines.txt       Access Level: general

company_docs/
  company_overview.txt               Access Level: public
  executive_strategy_fy2025.txt      Access Level: executive
  manager_handbook.txt               Access Level: manager

--------------------------------------------------------------------------------
SAMPLE QUERIES TO TEST WITH
--------------------------------------------------------------------------------

Use the following queries to validate your Cortex implementation and RBAC logic:

1. "What is the parental leave policy?"
   Expected: Works for ALL user tiers (public document)
   Tests: Basic retrieval, public access

2. "How do I request software approval?"
   Expected: Works for ALL user tiers (public document)
   Tests: IT policy retrieval, public access

3. "What are the FY2025 strategic priorities?"
   Expected: Works for EXEC tier ONLY (executive document)
   Tests: RBAC hard block — standard, general, and manager users should NOT
   receive this content; exec users should receive full detail

4. "What is the manager headcount plan for 2025?"
   Expected: Works for MANAGER and EXEC tiers
   Tests: RBAC partial block — standard users blocked; manager/exec users
   can see headcount freeze information from the manager handbook and
   executive strategy doc (exec only for the latter)

5. "How do I handle an underperforming team member?"
   Expected: Works for MANAGER and EXEC tiers
   Tests: Manager-level content retrieval from manager_handbook.txt

BONUS CHALLENGE QUERIES:

6. "What are the data encryption standards?"
   Expected: Works for STANDARD, MANAGER, EXEC (general access)
   Tests: General-tier access enforcement (should block anonymous users if
   you implement that tier)

7. "What is Meridian's stock ticker?"
   Expected: Works for ALL tiers (public company overview)
   Tests: Simple factual retrieval from public doc

8. "Who are the M&A acquisition targets?"
   Expected: EXEC tier retrieves the executive strategy doc, which notes
   names are REDACTED. A well-built RAG system should return this faithfully.
   Tests: Handling of intentionally incomplete/redacted information

--------------------------------------------------------------------------------
NOTES FOR STUDENTS
--------------------------------------------------------------------------------

- These documents are intentionally realistic but entirely fictional. "Meridian
  Technologies" does not exist; any resemblance to real companies is coincidental.

- Documents vary in length and structure to simulate real enterprise knowledge
  bases. Chunk sizes and overlap settings will affect retrieval quality —
  experiment with different values.

- The executive_strategy_fy2025.txt document is the only EXECUTIVE-level
  document. Use it to test that your RBAC filter correctly suppresses retrieval
  for non-exec users even when the query is highly relevant.

- The compensation_and_benefits.txt and data_security_guidelines.txt documents
  are GENERAL access (not public). If you implement a guest/anonymous tier,
  these should be blocked for unauthenticated users.

- Good luck, and build something great.

================================================================================
