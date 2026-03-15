import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from server.config import ServerConfig
from server.ws_manager import WSManager
from server.game_runner import GameRunner

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

config = ServerConfig()
ws_manager = WSManager()
game_runners: dict[str, GameRunner] = {}

DEFAULT_SESSION = "default"


@asynccontextmanager
async def lifespan(app: FastAPI):
    runner = GameRunner(DEFAULT_SESSION, ws_manager, config)
    game_runners[DEFAULT_SESSION] = runner
    await runner.start()
    logger.info("Default game session started")
    yield
    for sid, runner in game_runners.items():
        await runner.stop()
    logger.info("All sessions stopped")


app = FastAPI(title="NLE Agent Platform", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"status": "ok", "sessions": list(game_runners.keys())}


@app.get("/api/sessions")
async def list_sessions():
    results = []
    for sid, runner in game_runners.items():
        results.append({
            "session_id": sid,
            "running": runner.running,
            "spectators": ws_manager.spectator_count(sid),
        })
    return results


@app.websocket("/ws/spectate/{session_id}")
async def spectate_ws(websocket: WebSocket, session_id: str):
    if session_id not in game_runners:
        await websocket.close(code=4004, reason="Session not found")
        return
    await ws_manager.add_spectator(session_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.remove_spectator(session_id, websocket)
        logger.info("Spectator disconnected from session %s", session_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.api:app", host=config.host, port=config.port, reload=False)
