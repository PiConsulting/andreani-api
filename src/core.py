import os
from typing import List
from uuid import UUID
from pydantic import parse_obj_as
from job_process import Job, JobResponse
from os import PathLike
from queue import Queue
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi import UploadFile
from databases import Database
from azure.storage.blob import ContainerClient
from azure.storage.blob import BlobServiceClient


def save_upload_file_tmp(upload_file: UploadFile) -> Path:
    try:
        suffix = Path(upload_file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            tmp_path = Path(tmp.name)
    finally:
        upload_file.file.close()
    return tmp_path


def save_upload_file_storage(upload_file: UploadFile) -> Path:
    blob_service_client: BlobServiceClient = BlobServiceClient.from_connection_string(
        os.environ['CONNECTION_STR'])
    container_client: ContainerClient = blob_service_client.get_container_client(
        os.environ['CONTAINER_NAME'])
    try:
        blob_client = container_client.get_blob_client(upload_file.filename)
        blob_client.upload_blob(upload_file.file, blob_type="BlockBlob")
    finally:
        upload_file.file.close()
    return Path(upload_file.filename)


async def create_processing_video_job(db: Database, path: PathLike, name: str, job_queue: "Queue[Job]") -> UUID:
    print(path)
    job = Job(file_path=str(path), name = name)
    await job.create_on_db(db)
    job_queue.put(job)
    print("Done create_processing_video_job")
    return job.uid


async def get_job_list(db: Database) -> List[JobResponse]:
    query = "SELECT ROWID as id, * FROM jobs"
    jobs = await db.fetch_all(query=query)
    jobs = parse_obj_as(List[JobResponse], jobs)
    jobs.reverse()
    return jobs


def get_job_csv(job_uuid: str, csv_folder: str) -> str:
    file_path = os.path.join(csv_folder, f"{job_uuid}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist")
    return file_path
