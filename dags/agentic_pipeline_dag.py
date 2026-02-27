from airflow.sdk import dag, task, Param, get_current_context
import os
import logging

logger = logging.getLogger(__name__)


class Config:
    BASE_DIR = os.getenv(
        "PIPELINE_BASE_DIR", "/Users/fredoffei/Documents/selfHealingPipeline"
    )
    INPUT_FILE = os.getenv(
        "PIPELINE_INPUT_FILE", f"{BASE_DIR}/input/yelp_academic_dataset_review.json"
    )
    OUTPUT_FILE = os.getenv("PIPELINE_OUTPUT_FILE", f"{BASE_DIR}/output/")

    MAX_TEXT_LENGTH = int(os.getenv("PIPELINE_MAX_TEXT_LENGTH", 2000))
    DEFAULT_BATCH_SIZE = 100
    DEFAULT_OFFSET = 0
