# NLE LLM Agent Cloud Gaming Platform

NetHack Learning Environment + LLM Agent + Browser 3D Visualization

## Architecture

```
Cloud Server                          Browser
┌──────────────────┐    WebSocket     ┌────────────────────────┐
│  NLE Game Engine │◄───────────────►│  Three.js 3D Renderer  │
│  LLM Agent       │    Game State   │  React UI (HUD/Msg)    │
│  Memory System   │                 │  Agent Info Panel      │
│  FastAPI Server  │                 └────────────────────────┘
└──────────────────┘
```

## Project Structure

```
nle-agent-platform/
├── backend/
│   ├── env/              # NLE environment wrappers (score system)
│   ├── agent/            # LLM Agent (planner, executor, text observer)
│   ├── memory/           # Memory system (episode + note + reflection)
│   ├── server/           # FastAPI WebSocket server
│   ├── localization/     # Chinese localization engine
│   └── requirements.txt
├── frontend/             # Browser 3D viewer (forked from nethack-3d)
├── docs/                 # API documentation
└── sdk/                  # Python SDK for external agents
```

## Quick Start

```bash
# Backend
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn server.api:app --host 0.0.0.0 --port 8000

# Frontend
cd frontend
npm install
npm run dev
```

## License

MIT
