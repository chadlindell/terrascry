from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import logging
from pathlib import Path
from daemon import HirtDaemon, ScanConfig  # Import our backend

app = FastAPI(title="HIRT Field Controller")

# Global State (in memory for Phase 1)
daemon = HirtDaemon(data_dir="data_out", mock_mode=True)
current_scan_file = None

class ScanRequest(BaseModel):
    survey_name: str
    operator: str
    site_id: str

@app.post("/api/scan/start")
async def start_scan(req: ScanRequest):
    global current_scan_file
    try:
        config = ScanConfig(
            survey_name=req.survey_name,
            operator=req.operator,
            site_id=req.site_id
        )
        # In a real app, we'd spawn the daemon loop in a background thread
        # For now, we just create the file to prove the link
        current_scan_file = daemon.start_new_survey(config)
        return {"status": "started", "file": str(current_scan_file)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/scan/status")
async def get_status():
    if current_scan_file:
        return {
            "status": "running", 
            "samples": daemon.dset_mit.shape[0] if daemon.current_h5 else 0,
            "file": str(current_scan_file)
        }
    return {"status": "idle"}

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

# Create static dir for the frontend HTML
Path("static").mkdir(exist_ok=True)
