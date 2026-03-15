from pathlib import Path
"""FastAPI application with REST + WebSocket endpoints for multi-session NLE platform."""
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

from server.config import ServerConfig
from server.ws_manager import WSManager
from server.game_runner import GameRunner
from server.session_manager import SessionManager
from server.nle_event_converter import NLEEventConverter
from agent.action_translator import NLEActionTranslator
from agent.text_observer import NLETextObserver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

config = ServerConfig()
ws_manager = WSManager()
session_manager = SessionManager()
game_runners: dict[str, GameRunner] = {}

DEFAULT_SESSION = "default"


# ── Pydantic models ──────────────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    agent_name: str = "anonymous"
    config: dict = {}


class CreateSessionResponse(BaseModel):
    session_id: str
    agent_token: str
    spectate_url: str


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    runner = GameRunner(DEFAULT_SESSION, ws_manager, config)
    game_runners[DEFAULT_SESSION] = runner
    await runner.start()
    logger.info("Default demo session started")
    yield
    for sid, runner in game_runners.items():
        await runner.stop()
    session_manager.cleanup_all()
    logger.info("All sessions stopped")


app = FastAPI(title="NLE Agent Platform", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST endpoints ───────────────────────────────────────────────────────────



@app.get("/")
async def root():
    return RedirectResponse("/static/dashboard.html")
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "active_sessions": len(session_manager.list_sessions()),
        "demo_session": DEFAULT_SESSION,
    }


@app.post("/api/sessions", response_model=CreateSessionResponse)
async def create_session(req: CreateSessionRequest):
    try:
        result = session_manager.create_session(req.agent_name, req.config)
    except RuntimeError as e:
        raise HTTPException(status_code=429, detail=str(e))
    return result


@app.get("/api/sessions")
async def list_sessions():
    managed = session_manager.list_sessions()
    demo_info = []
    for sid, runner in game_runners.items():
        if not any(s["session_id"] == sid for s in managed):
            demo_info.append({
                "session_id": sid,
                "agent_name": "demo",
                "status": "running" if runner.running else "idle",
                "spectators": ws_manager.spectator_count(sid),
            })
    return {"sessions": demo_info + managed}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    if session_id == DEFAULT_SESSION:
        runner = game_runners.get(DEFAULT_SESSION)
        if runner:
            return {
                "session_id": DEFAULT_SESSION,
                "agent_name": "demo",
                "status": "running" if runner.running else "idle",
                "spectators": ws_manager.spectator_count(DEFAULT_SESSION),
            }
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.summary()


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id == DEFAULT_SESSION:
        raise HTTPException(status_code=403, detail="Cannot delete demo session")
    runner = game_runners.pop(session_id, None)
    if runner:
        await runner.stop()
    deleted = session_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


# ── WebSocket: Spectator ─────────────────────────────────────────────────────

@app.websocket("/ws/spectate/{session_id}")
async def spectate_ws(websocket: WebSocket, session_id: str):
    has_session = (
        session_id in game_runners
        or session_manager.get_session(session_id) is not None
    )
    if not has_session:
        await websocket.close(code=4004, reason="Session not found")
        return
    await ws_manager.add_spectator(session_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.remove_spectator(session_id, websocket)
        logger.info("Spectator disconnected from session %s", session_id)


# ── WebSocket: Agent ─────────────────────────────────────────────────────────

@app.websocket("/ws/agent/{session_id}")
async def agent_ws(websocket: WebSocket, session_id: str, token: str = Query(...)):
    session = session_manager.get_session(session_id)
    if session is None or session.agent_token != token:
        await websocket.close(code=4001, reason="Unauthorized or session not found")
        return

    await websocket.accept()
    session.status = session.status.__class__("running")
    session.touch()

    translator = NLEActionTranslator()
    observer = NLETextObserver()

    try:
        obs, info = session.env.reset()
        session.converter.reset()

        vis_events = session.converter.obs_to_events(obs, full=True)
        ws_manager.cache_full_state(session_id, vis_events)
        for ev in vis_events:
            await ws_manager.broadcast_to_spectators(session_id, ev)

        text_obs = observer.observe(obs, info)
        await websocket.send_json({
            "type": "observation",
            "observation": text_obs,
            "meta": info,
            "available_actions": NLEActionTranslator.available_actions(),
            "done": False,
        })

        while True:
            raw = await websocket.receive_json()
            session.touch()
            msg_type = raw.get("type", "action")

            if msg_type == "action":
                text_action = raw.get("action", "wait")
                action_indices = translator.translate(text_action)

                last_obs, last_info, total_reward, done = obs, info, 0.0, False
                for act_idx in action_indices:
                    last_obs, reward, done, truncated, last_info = session.env.step(act_idx)
                    session.total_steps += 1
                    total_reward += reward

                    vis_events = session.converter.obs_to_events(
                        last_obs, reward=reward, done=done,
                        full=(session.total_steps % 50 == 0),
                    )
                    for ev in vis_events:
                        await ws_manager.broadcast_to_spectators(session_id, ev)

                    if done:
                        break

                text_obs = observer.observe(last_obs, last_info)
                session.memory.record_observation(
                    content=f"Action: {text_action} -> {text_obs.get('message', '')}",
                    step_number=session.total_steps,
                )

                await websocket.send_json({
                    "type": "observation",
                    "observation": text_obs,
                    "meta": last_info,
                    "available_actions": NLEActionTranslator.available_actions(),
                    "done": done,
                    "reward": total_reward,
                })
                obs, info = last_obs, last_info

                if done:
                    await websocket.send_json({
                        "type": "game_over",
                        "meta": last_info,
                    })

            elif msg_type == "plan":
                plan = raw.get("plan", {})
                actions = plan.get("actions", [])
                session.memory.record_plan(
                    content=str(plan),
                    step_number=session.total_steps,
                )
                results = []
                for text_action in actions:
                    action_indices = translator.translate(text_action)
                    for act_idx in action_indices:
                        obs, reward, done, truncated, info = session.env.step(act_idx)
                        session.total_steps += 1

                        vis_events = session.converter.obs_to_events(
                            obs, reward=reward, done=done,
                            full=(session.total_steps % 50 == 0),
                        )
                        for ev in vis_events:
                            await ws_manager.broadcast_to_spectators(session_id, ev)

                        if done:
                            break

                    text_obs = observer.observe(obs, info)
                    results.append({
                        "action": text_action,
                        "observation": text_obs,
                        "done": done,
                        "reward": reward,
                    })
                    if done:
                        break

                await websocket.send_json({
                    "type": "plan_result",
                    "results": results,
                    "meta": info,
                    "available_actions": NLEActionTranslator.available_actions(),
                    "done": done,
                })

    except WebSocketDisconnect:
        logger.info("Agent disconnected from session %s", session_id)
    except Exception as e:
        logger.error("Agent WS error in session %s: %s", session_id, e, exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        session.status = session.status.__class__("idle")


# ── Static files ─────────────────────────────────────────────────────────────

_frontend_dist = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "frontend", "dist"
)
if os.path.isdir(_frontend_dist):
    app.mount("/frontend", StaticFiles(directory=_frontend_dist, html=True), name="frontend")

_static_dir = Path(__file__).parent.parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir), html=True), name="static")
    logger.info("Serving frontend from %s", _frontend_dist)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.api:app", host=config.host, port=config.port, reload=False)
