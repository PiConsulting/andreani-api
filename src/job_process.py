from queue import Queue
from threading import Thread
from typing import List, Optional
from uuid import UUID, uuid4
from databases.core import Database
from pydantic import BaseModel, Field
from pydantic import parse_obj_as
from enum import Enum
import asyncio
import ml_engine
from fastapi.encoders import jsonable_encoder
import os
import traceback


class JobStatusEnum(str, Enum):
    in_progress = "in_progress"
    new = "new"
    done = "done"
    error = "error"


class Job(BaseModel):
    name: Optional[str] = "ejemplo"
    uid: UUID = Field(default_factory=uuid4)
    status: JobStatusEnum = JobStatusEnum.new
    file_path: str

    async def create_on_db(self, db: Database):
        query = "INSERT INTO jobs(uid, status, file_path, name) VALUES (:uid, :status, :file_path, :name)"
        values = jsonable_encoder(self)
        await db.execute(query=query, values=values)

    async def update_status(self, db: Database, status: JobStatusEnum):
        self.status = status
        query = "UPDATE jobs SET status = :status WHERE uid = :uid"
        values = {"uid": str(self.uid), "status": self.status}
        await db.execute(query=query, values=values)


class JobResponse(Job):
    id: Optional[int] = 0


class JobProcessor(Thread):

    def __init__(self, q: "Queue[Job]", db: Database, csv_folder: str):
        Thread.__init__(self, daemon=True)
        self.q = q
        self.db = db
        self.csv_folder = csv_folder
        os.makedirs(self.csv_folder, exist_ok=True)

    async def grab_jobs_from_db(self):
        print("Grabbing 'old' pending jobs")
        query = "SELECT * FROM jobs where status = :status"
        pending_jobs = await self.db.fetch_all(query, values={"status": JobStatusEnum.new})
        pending_jobs = parse_obj_as(List[Job], pending_jobs)
        for job in pending_jobs:
            self.q.put(job)

    def process_job(self, job: Job):
        try:
            asyncio.run(job.update_status(self.db, JobStatusEnum.in_progress))
            print("processing job", job)
            # result: List[str] = ml_engine.run_inference(job.file_path)

            ml_engine.run_job(job.file_path, f"{job.uid}.csv")
            print("done with inference")

            # csv_filename = f"{job.uid}.csv"
            # pd.DataFrame(data={'code': result}).to_csv(os.path.join(self.csv_folder, csv_filename), index=False)
            print("done processing job")
            status = JobStatusEnum.done
        except Exception as err:
            traceback.print_exc()
            status = JobStatusEnum.error

        asyncio.run(job.update_status(self.db, status))

    def run(self):
        while True:
            job = self.q.get()
            print("processing queue")
            self.process_job(job)
            print("done processing queue")
            self.q.task_done()
