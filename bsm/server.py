"""
server.py — Local REST API Microservice for ABM / BSM Agent Memory.

Exposes HTTP endpoints for memory storage, zero-cost semantic routing,
multi-hop query reasoning, memory contracts, and persistence.
"""

from typing import Optional, List, Dict, Any
from bsm.integrations.agent_memory import AgentMemory

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    BaseModel = object


# Global agent memory instance
memory_instance = AgentMemory(dim=2048)

if HAS_FASTAPI:
    app = FastAPI(
        title="ABM / BSM Agent Memory Service",
        description="Local microsecond-latency deterministic memory and pre-LLM semantic router.",
        version="2.0.0"
    )

    class RememberRequest(BaseModel):
        subject: Optional[str] = None
        relation: Optional[str] = None
        object: Optional[str] = None
        text: Optional[str] = None
        weight: int = 1

    class RecallRequest(BaseModel):
        subject: str
        relation: str

    class SemanticRouteRequest(BaseModel):
        prompt: str
        entity: str
        relation: str

    class PersistenceRequest(BaseModel):
        path: str

    @app.post("/remember")
    def api_remember(req: RememberRequest):
        if req.text:
            res = memory_instance.remember_text(req.text)
            return {"status": "ok", "stored": res}
        elif req.subject and req.relation and req.object:
            res = memory_instance.remember(req.subject, req.relation, req.object, weight=req.weight)
            return {"status": "ok", "stored": res}
        else:
            raise HTTPException(status_code=400, detail="Must provide (subject, relation, object) or text.")

    @app.post("/recall")
    def api_recall(req: RecallRequest):
        ans, conf = memory_instance.recall(req.subject, req.relation)
        return {"answer": ans, "confidence": conf, "found": ans is not None}

    @app.post("/semantic_route")
    def api_semantic_route(req: SemanticRouteRequest):
        res = memory_instance.semantic_route(req.prompt, req.entity, req.relation)
        return {
            "handled": res.handled,
            "answer": res.answer,
            "confidence": res.confidence,
            "source": res.source,
            "forward_to_llm": res.forward_to_llm,
            "explanation": res.explanation
        }

    @app.get("/stats")
    def api_stats():
        return memory_instance.stats()

    @app.post("/save")
    def api_save(req: PersistenceRequest):
        memory_instance.save(req.path)
        return {"status": "ok", "saved_to": req.path}

    @app.post("/load")
    def api_load(req: PersistenceRequest):
        global memory_instance
        memory_instance = AgentMemory.load(req.path)
        return {"status": "ok", "loaded_from": req.path, "stats": memory_instance.stats()}
else:
    app = None


def run_server(host: str = "127.0.0.1", port: int = 8000):
    if not HAS_FASTAPI or app is None:
        print("FastAPI not installed. Please install with: pip install fastapi uvicorn")
        return
    try:
        import uvicorn
        uvicorn.run(app, host=host, port=port)
    except ImportError:
        print("uvicorn not installed. Please install with: pip install uvicorn")



if __name__ == "__main__":
    run_server()
