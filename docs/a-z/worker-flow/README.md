# Queue + Worker Flow

## 1. Queueing
- API creates an `inference_jobs` row with status `queued`.
- API enqueues `app.tasks.run_inference_job` to Redis via RQ.

## 2. Worker execution
- Worker process (`apps/api/worker.py`) listens on queue `legal-risk`.
- Worker claims queued jobs and updates status to `running`.

## 3. Inference lifecycle
- Worker loads contract text by `contract_id`.
- Runs classifier inference flow.
- Stores result JSON and marks job `succeeded`.

## 4. Failure handling
- On exceptions, worker marks job `failed` and writes `error_message`.
- Error is surfaced through `GET /jobs/{id}`.

