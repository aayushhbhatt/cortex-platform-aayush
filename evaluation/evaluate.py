from __future__ import annotations
import argparse, json, math, os
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import sys
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.pipeline import retrieve
load_dotenv()
@dataclass
class EvaluationCase:
    query:str; user_tier:str; relevant_doc_ids:list[str]; relevant_chunk_id_prefixes:list[str]; notes:str=""

def load_golden_dataset(path: str | Path = "evaluation/golden_dataset.json") -> list[EvaluationCase]:
    payload=json.loads(Path(path).read_text(encoding='utf-8'))
    return [EvaluationCase(query=i['query'],user_tier=i['user_tier'],relevant_doc_ids=i.get('relevant_doc_ids',[]),relevant_chunk_id_prefixes=i.get('relevant_chunk_id_prefixes',[]),notes=i.get('notes','')) for i in payload]

def is_relevant_result(result, case: EvaluationCase) -> bool:
    doc_id=getattr(result,'doc_id','') or ''; chunk_id=getattr(result,'chunk_id','') or ''
    return doc_id in case.relevant_doc_ids or any(chunk_id.startswith(p) for p in case.relevant_chunk_id_prefixes)

def precision_at_k(results, case: EvaluationCase, k: int = 5) -> float:
    if k<=0:return 0.0
    return sum(1 for r in results[:k] if is_relevant_result(r, case))/k

def reciprocal_rank(results, case: EvaluationCase) -> float:
    for idx,r in enumerate(results,1):
        if is_relevant_result(r,case): return 1.0/idx
    return 0.0

def dcg_at_k(relevance:list[int],k:int=5)->float:
    return sum(rel/math.log2(rank+1) for rank,rel in enumerate(relevance[:k],1))

def ndcg_at_k(results, case: EvaluationCase, k: int = 5) -> float:
    if k<=0:return 0.0
    rel=[1 if is_relevant_result(r,case) else 0 for r in results[:k]]; ideal=sorted(rel, reverse=True)
    idcg=dcg_at_k(ideal,k); return 0.0 if idcg==0 else dcg_at_k(rel,k)/idcg

def evaluate_retrieval(dataset_path:str|Path='evaluation/golden_dataset.json',data_dir:str|Path='data',registry_db_path:str|Path|None='data/cortex_registry.sqlite',top_k:int=5,user_tier:str='standard',use_pgvector:bool=True,database_url:str|None=None)->dict:
    cases=load_golden_dataset(dataset_path); data_path=Path(data_dir)
    if not data_path.exists(): raise FileNotFoundError(f"Data directory '{data_path}' does not exist. Please provide a valid data_dir containing .txt files.")
    per_case=[]; p=[]; rr=[]; nd=[]
    for case in cases:
        effective_tier=user_tier or case.user_tier
        results=retrieve(query=case.query,top_k=top_k,data_dir=data_path,user_tier=effective_tier,registry_db_path=registry_db_path,use_pgvector=True,database_url=database_url)
        pa=precision_at_k(results,case,top_k); r=reciprocal_rank(results,case); n=ndcg_at_k(results,case,top_k)
        p.append(pa); rr.append(r); nd.append(n)
        per_case.append({"query":case.query,"user_tier":effective_tier,"retrieved_doc_ids":[x.doc_id for x in results],"retrieved_chunk_ids":[x.chunk_id for x in results],"precision_at_k":pa,"reciprocal_rank":r,"ndcg_at_k":n})
    c=len(cases)
    return {"case_count":c,"top_k":top_k,"precision_at_k":(sum(p)/c) if c else 0.0,"mrr":(sum(rr)/c) if c else 0.0,"ndcg_at_k":(sum(nd)/c) if c else 0.0,"cases":per_case,"retrieval_config":{"data_dir":str(data_dir),"registry_db_path":str(registry_db_path) if registry_db_path else None,"top_k":top_k,"user_tier":user_tier,"use_pgvector":True,"database_url_configured":bool(database_url),"rag_embedding_provider":"openai","rag_chunk_strategy":os.getenv('RAG_CHUNK_STRATEGY','word_window')}}

def _parse_args():
    p=argparse.ArgumentParser();
    p.add_argument('--dataset-path',default='evaluation/golden_dataset.json');p.add_argument('--data-dir',default='data');p.add_argument('--registry-db-path',default='data/cortex_registry.sqlite')
    p.add_argument('--top-k',type=int,default=5);p.add_argument('--user-tier',choices=['standard','manager','exec'],default='standard');p.add_argument('--database-url',default=None)
    p.add_argument('--mode',choices=['local_hash_word_window','local_hash_section','openai_pgvector_section']);p.add_argument('--output-path',default=None); return p.parse_args()

if __name__=='__main__':
    args=_parse_args()
    expected={}
    if args.mode=='local_hash_word_window': expected={"RAG_EMBEDDING_PROVIDER":"openai","RAG_CHUNK_STRATEGY":"word_window","RAG_USE_PGVECTOR":"true"}
    elif args.mode=='local_hash_section': expected={"RAG_EMBEDDING_PROVIDER":"openai","RAG_CHUNK_STRATEGY":"section","RAG_USE_PGVECTOR":"true"}
    elif args.mode=='openai_pgvector_section': expected={"RAG_EMBEDDING_PROVIDER":"openai","RAG_CHUNK_STRATEGY":"section","RAG_USE_PGVECTOR":"true"}
    summary=evaluate_retrieval(dataset_path=args.dataset_path,data_dir=args.data_dir,registry_db_path=args.registry_db_path,top_k=args.top_k,user_tier=args.user_tier,use_pgvector=True,database_url=args.database_url)
    print('Retrieval Evaluation Summary'); print(f"Case count: {summary['case_count']}"); print(f"P@{summary['top_k']}: {summary['precision_at_k']:.4f}"); print(f"MRR: {summary['mrr']:.4f}"); print(f"NDCG@{summary['top_k']}: {summary['ndcg_at_k']:.4f}")
    print('retrieval_config:', json.dumps(summary['retrieval_config'], indent=2))
    if expected: print('expected_mode_env:', json.dumps(expected))
    if args.output_path:
        out=Path(args.output_path)
    else:
        out=Path('evaluation/results')/f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{args.mode or 'default'}.json"
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(f"Saved: {out}")
