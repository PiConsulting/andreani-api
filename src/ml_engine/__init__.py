import os
import time
import requests
from typing import List
from .video_stream import process_video


def run_inference(video_path: str, debug=False):
    found_codes: List[str] = []
    for text in process_video(video_path, debug=debug):
        if text in found_codes:
            print("Code already in list:", text)
        else:
            print("New code:", text )
            found_codes.append(text)
    return found_codes


def run_job(filepath, job_uuid):
    body = {"job_id": 1,
            "notebook_params": {"storage_filepath": filepath,
                                "storage_jobname": job_uuid}
           }

    response = requests.post(
        os.environ['DBRICKS_URL'] + 'run-now',
        headers={'Authorization': 'Bearer %s' % os.environ['DBRICKS_TOKEN']},
        json=body
    )
    if response.status_code == 200:
        response = response.json()
        run_id = response.get('run_id', 0)
        while True:
            response = requests.get(
                os.environ['DBRICKS_URL'] + f'runs/get-output?run_id={run_id}',
                headers={'Authorization': 'Bearer %s' % os.environ['DBRICKS_TOKEN']}
            )
            if response.status_code == 200:
                response = response.json()
                if response['metadata']['state'].get('result_state', '') == "SUCCESS":
                    csv_path = response['notebook_output']['result']
                    return csv_path
                if response['metadata']['state'].get('life_cycle_state', '') == "TERMINATED":
                    raise Exception('fail dbricks process')
            time.sleep(5)
    raise Exception('fail on check job status')
