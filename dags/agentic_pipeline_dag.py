from datetime import timedelta
import datetime
import itertools
import json
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

    # OLLAMA SETTINGS
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", " llama3.2")
    OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", 120))
    OLLAMA_RETRIES = int(os.getenv("OLLAMA_RETRIES", 3))


default_args = {
    "owner": "cridix",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=30),
}


def _load_ollama_model(model_name: str):
    import ollama

    logger.info(f"Loading Ollama model: {model_name}")
    logger.info(f"Ollama host: {Config.OLLAMA_HOST}")

    client = ollama.Ollama(host=Config.OLLAMA_HOST)

    try:
        client.show(model_name)
        logger.info(f"OLLAMA model {model_name} is available")
    except ollama.ResponseError as e:
        logger.info(
            "Model not found locally, Attempting to pull form remote repository..."
        )
        try:
            client.pull(model_name)
            logger.info(f"OLLAMA model {model_name} pulled successfully")
        except ollama.ResponseError as pull_error:
            logger.error(f"Failed to pull OLLAMA model {model_name}: {pull_error}")
            raise

    test_response = client.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": "Classify the sentiment: 'This is a great product!' as positive, negative, or neutral.",
            }
        ],
    )

    test_results = test_response["message"]["content"].strip().upper()
    logger.info(f"Model validation passed with test response: {test_results}")

    return {
        "backend": "ollama",
        "model_name": model_name,
        "ollama_host": Config.OLLAMA_HOST,
        "max_length": Config.MAX_TEXT_LENGTH,
        "status": "loaded",
        "validated_at": datetime.now().isoformat(),
    }


def _load_from_file(params: dict, batch_size: int, offset: int):
    input_file = params.get("input_file", Config.INPUT_FILE)

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    reviews = []

    with open(input_file, "r", encoding="utf-8") as f:
        sliced = itertools.islice(f, offset, offset + batch_size)

        for line in sliced:
            try:
                reviews = json.load(line.strip())
                reviews.append(
                    {
                        "review_id": review.get("review_id"),
                        "business_id": review.get("business_id"),
                        "user_id": review.get("user_id"),
                        "stars": review.get("stars", 0),
                        "text": review.get("text"),
                        "date": review.get("date"),
                        "useful": review.get("useful", 0),
                        "funny": review.get("funny", 0),
                        "cool": review.get("cool", 0),
                    }
                )
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON line: {e}")
                continue

    logger.info(
        f"Loaded {len(reviews)} reviews from file starting from offset {offset}."
    )
    return reviews


def _parse_ollama_response(reponse_text: str):
    try:
        clean_text = response_text.strip()

        if clean_text.startswith("```"):
            lines = clean_text.split("\n")
            clean_text = (
                "\n".join(lines[1:-1]) if lines[-1] == "```" else "\n".join(lines[1:])
            )

            parsed = json.loads(clean_text)
            sentiment = parsed.get("sentiment", "NEUTRAL").upper()
            confidence = float(parsed.get("confidence", 0.0))

            if sentiment not in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
                sentiment = "NEUTRAL"

            return {"label": sentiment, "score": min(max(confidence, 0.0), 1.0)}
    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:

        upper_text = response_text.strip().upper()
        if "POSITIVE" in upper_text:
            return {"label": "POSITIVE", "score": 0.75}
        elif "NEGATIVE" in upper_text:
            return {"label": "NEGATIVE", "score": 0.75}

        return {"label": "NEUTRAL", "score": 0.50}


@dag(
    dag_id="self_healing_pipeline",
    default_args=default_args,
    description="Pipeline for sentiment analysis using OLLAMA model",
    schedule=None,
    start_date=datetime(2026, 2, 27),
    catchup=False,
    tags=["sentiment_analysis", "nlp", "ollama", "yelp_reviews"],
    params={
        "input_file": Param(
            default=Config.INPUT_FILE,
            type="string",
            description="Path to the input JSON file containing Yelp reviews",
        ),
        "batch_size": Param(
            default=Config.DEFAULT_BATCH_SIZE,
            type="integer",
            description="Number of reviews to process in each batch",
        ),
        "offset": Param(
            default=Config.DEFAULT_OFFSET,
            type="integer",
            description="Offset to start reading reviews from the input file.",
        ),
        "ollama_model": Param(
            default=Config.OLLAMA_MODEL,
            type="string",
            description="Name of the OLLAMA model to use for sentiment analysis.",
        ),
    },
    render_template_as_native_obj=True,
)
def self_healing_pipeline():
    # Implement the self-healing logic here
    @task()
    def load_model():
        context = get_current_context()
        params = context["params"]
        model_name = params.get("ollama_model", Config.OLLAMA_MODEL)
        logger.info(f"Using OLLAMA model: {model_name}")
        return _load_ollama_model(model_name)

    @task()
    def load_reviews():
        context = get_current_context()
        params = context["params"]
        batch_size = params.get("batch_size", Config.DEFAULT_BATCH_SIZE)
        offset = params.get("offset", Config.DEFAULT_OFFSET)
        logger.info(
            f"Loading reviews with batch size: {batch_size} and offset: {offset}"
        )
        return _load_from_file(params, batch_size, offset)
