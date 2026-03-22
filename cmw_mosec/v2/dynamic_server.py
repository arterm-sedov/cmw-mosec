"""Dynamic server - Mosec standard pattern.

Port is handled by Mosec CLI automatically:
- CLI: python -m cmw_mosec.v2.dynamic_server --port 8000
- Env: MOSEC_PORT=8000 python -m cmw_mosec.v2.dynamic_server
"""

import os

from mosec import Runtime, Server

from .workers import (
    EmbeddingWorkerV2,
    GuardWorkerV2,
    RerankWorkerV2,
    ScoreWorkerV2,
)


def run_server():
    """Start server with runtime-configured workers."""
    server = Server()

    routes = {}

    if os.getenv("ACTIVE_EMBEDDING_MODEL"):
        routes["/v1/embeddings"] = [Runtime(EmbeddingWorkerV2)]

    if os.getenv("ACTIVE_RERANKER_MODEL"):
        routes["/v1/score"] = [Runtime(ScoreWorkerV2)]
        routes["/v1/rerank"] = [Runtime(RerankWorkerV2)]

    if os.getenv("ACTIVE_GUARD_MODEL"):
        routes["/v1/moderate"] = [Runtime(GuardWorkerV2)]

    server.register_runtime(routes)
    server.run()


if __name__ == "__main__":
    run_server()
