from celery import Celery

# Redis acts as the Broker (queue) and Backend (result storage)
celery = Celery(
    "worker",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

@celery.task(bind=True)
def process_pdf_task(self, file_path):
    # Your miner_pdf_to_md logic goes here
    # Use self.update_state to send progress updates if you want!
    return {"status": "completed", "path": "output.md"}