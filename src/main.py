import os
from queue import Queue
from typing import List
from uuid import UUID
from databases import Database
from pathlib import Path

from dotenv import load_dotenv

from fastapi import (APIRouter, Depends, FastAPI, File, HTTPException,
                     UploadFile, Form)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from auth import get_current_account
from auth import router as auth_router
from core import (create_processing_video_job, get_job_csv, get_job_list,
                  save_upload_file_storage)
from job_process import JobProcessor, JobResponse


env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)


CSV_FOLDER = "../results/"
DOWNLOAD_CSV_NAME = "codigos.csv"
SQLITE_PATH = "../sqlite.db"

if not os.path.exists(CSV_FOLDER):
    os.mkdir(CSV_FOLDER)

app = FastAPI()
app.include_router(auth_router)
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])
app.add_middleware(TrustedHostMiddleware,
                   allowed_hosts=["*.z20.web.core.windows.net",
                                  "*.azurewebsites.net",
                                  "localhost",
                                  "127.0.0.1",
                                  "0.0.0.0"])
hello_router = APIRouter()
router = APIRouter(dependencies=[Depends(get_current_account)])


job_queue = Queue()


#DATABASE
db_existed = os.path.exists(SQLITE_PATH)
database = Database(f'sqlite:///{SQLITE_PATH}')


async def get_db():
    try:
        await database.connect()
        yield database
    finally:
        await database.disconnect()


class UploadVideoResponse(BaseModel):
    job_uid: str
    path: str


@hello_router.get("/")
async def hello():
    return "API Andreani"


@router.post("/upload-video", response_model=UploadVideoResponse, dependencies=[Depends(get_current_account)])
async def create_upload_file(file: UploadFile = File(...), name: str = Form(...), db: Database = Depends(get_db)):
    # path = save_upload_file_tmp(file)
    path = save_upload_file_storage(file)
    job_uid = await create_processing_video_job(db, path, name, job_queue)
    return {"path": str(path), "job_uid": str(job_uid)}


class JobsResponse(BaseModel):
    jobs: List[JobResponse]


@router.get("/jobs", response_model=JobsResponse, description="Returns the job list, new jobs appear first")
async def get_jobs(db: Database = Depends(get_db)):
    return {"jobs": await get_job_list(db)}


@router.get("/jobs/{job_uiid}")
async def get_job(job_uiid: UUID, db: Database = Depends(get_db)):
    try:
        path = get_job_csv(str(job_uiid), CSV_FOLDER)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job CSV not found")
    return FileResponse(path, media_type="octet/stream", filename=DOWNLOAD_CSV_NAME)


@app.on_event("startup")
async def startup_event():
    try:
        await database.connect()
        if not db_existed:
            # We have to initialize the DB tables
            with open("init.sql", 'r') as file:
                queries = file.read().split(";")
                for q in queries:
                    await database.execute(q)
        job_processor = JobProcessor(job_queue, database, CSV_FOLDER)
        job_processor.start()
        await job_processor.grab_jobs_from_db()
    finally:
        await database.disconnect()


app.include_router(router)
app.include_router(hello_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
