.PHONY: dev backend frontend install

# Start both backend and frontend
dev: backend frontend

# Start FastAPI backend on port 8100
backend:
	cd .. && python3 -m uvicorn dashboard.backend.main:app --port 8100 --reload &

# Start Vite frontend on port 5173
frontend:
	cd frontend && npm run dev &

# Install dependencies
install:
	pip3 install --break-system-packages fastapi uvicorn
	cd frontend && npm install
