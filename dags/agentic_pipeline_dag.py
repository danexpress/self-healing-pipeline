from datetime import timedelta
import datetime
import itertools
import json
import re
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


def _heal_review(review: dict) -> dict:
    text = review.get("text", "")

    result = {
        "review_id": review.get("review_id"),
        "business_id": review.get("business_id"),
        "stars": review.get("stars", 0),
        "original_text": None,
        "error_type": None,
        "action_taken": "none",
        "was_healed": False,
        "metadata": {
            "user_id": review.get("user_id"),
            "date": review.get("date"),
            "useful": review.get("useful", 0),
            "funny": review.get("funny", 0),
            "cool": review.get("cool", 0),
        },
    }

    if isinstance(text, (str, int, float, bool, type(None))):
        result["original_text"] = text
    else:
        result["original_text"] = str(text) if text else None

    if text is None:
        result["error_type"] = "missing_text"
        result["action_taken"] = "filled_with_placeholder"
        result["healed_text"] = "No review text provided."
        result["was_healed"] = True
        return result
    elif not isinstance(text, str):
        result["error_type"] = "wrong_type"
        try:
            converted = str(text).strip()
            result["healed_text"] = (
                converted if converted else "No review text provided."
            )
        except Exception as e:
            result["healed_text"] = "Conversion failed."

        result["action_taken"] = "type_conversion"
        result["was_healed"] = True
    elif not text.strip():
        result["error_type"] = "empty_text"
        result["action_taken"] = "filled_with_placeholder"
        result["healed_text"] = "No review text provided."
        result["was_healed"] = True
    elif not re.search(r"[A-Za-z0-9]", text):
        result["error_type"] = "special_characters_only"
        result["healed_text"] = "[Non-text content]"
        result["action_taken"] = "replaced_special_characters"
        result["was_healed"] = True
    elif len(text) > Config.MAX_TEXT_LENGTH:
        result["error_type"] = "text_too_long"
        result["healed_text"] = text[: Config.MAX_TEXT_LENGTH - 3] + "..."
        result["action_taken"] = "truncated_text"
        result["was_healed"] = True
    else:
        result["healed_text"] = text.strip()
        result["was_healed"] = False

    return result


def _analyzed_with_ollama(handled_reviews: list[dict], model_info: dict) -> list[dict]:
    import ollama
    import time

    model_name = model_info.get("model_name")
    ollama_host = model_info.get("ollama_host", Config.OLLAMA_HOST)

    try:
        client = ollama.Client(host=ollama_host)
    except Exception as e:
        logger.error(f"Failed to connect to OLLAMA at {ollama_host}: {e}")
        return _created_degraded_results(handled_reviews, str(e))

    results = []
    total = len(healed_reviews)

    for idx, review in enumerate(healed_reviews):
        text = review.get("healed_text", "")
        prediction = None

        for attempt in range(Config.OLLAMA_RETRIES):
            try:
                prompt = f""" 
                    Analyze the sentiment of this review and classify it as  POSITIVE, NEGATIVE, or NEUTRAL.
                    Review: "{text}"
                    Reply with ONLY a JSON object: {{"sentiment": "POSITIVE", "confidence": 0.95}}
                    """

                response = client.chat(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.1},
                )

                response_text = response["message"]["content"].strip()
                prediction = _parse_ollama_response(response_text)
                break
            except Exception as e:
                if attempt < Config.OLLAMA_RETRIES - 1:
                    logger.warning(
                        f"Attempt {attempt + 1} failed for review {review['review_id']}: {e}. Retrying..."
                    )
                    time.sleep(1)
                else:
                    logger.error(
                        f"All attempts failed for review {review['review_id']}: {e}"
                    )
                    prediction = {"label": "NEUTRAL", "score": 0.5, "error": str(e)}

        if (idx + 1) % 10 == 0 or idx == total:
            logger.info(f"Processed {idx + 1}/{total} reviews for sentiment analysis.")

        results.append(
            {
                "review_id": review.get("review_id"),
                "business_id": review.get("business_id"),
                "stars": review.get("stars", 0),
                "text": review.get("healed_text", ""),
                "original_text": review.get("original_text", ""),
                "predicted_sentiment": prediction.get("label"),
                "confidence": round(prediction.get("score"), 4),
                "status": "healed" if review.get("was_healed") else "success",
                "healing_applied": review.get("was_healed"),
                "healing_actions": (
                    review.get("action_taken") if review.get("was_healed") else None
                ),
                "error_type": (
                    review.get("error_type") if review.get("was_healed") else None
                ),
                "metadata": review.get("metadata", {}),
            }
        )
    logger.info(f"Ollama inference complete: {len(results)}/{total} reviews processed.")
    return results


def _created_degraded_results(
    handled_reviews: list[dict], error_message: str
) -> list[dict]:
    return [
        {
            **review,
            "text": review.get("healed_text", ""),
            "predicted_sentiment": "NEUTRAL",
            "confidence": 0.5,
            "status": "degraded",
            "error_message": error_message,
        }
        for review in handled_reviews
    ]


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

    @task()
    def diagnose_and_heal_batch(reviews: list[dict]):
        healed_reviews = [_heal_review(review) for review in reviews]
        heal_count = sum(1 for r in healed_reviews if r.get("was_healed", True))
        logger.info(f"Healed {heal_count} out of {len(reviews)} reviews in the batch.")
        return healed_reviews

    @task()
    def analyze_sentiment_batch(healed_reviews: list[dict], model_info: dict):
        if not healed_reviews:
            return []
        logger.info(f"Analyzing {len(healed_reviews)} reviews for sentiment.")
        return _analyzed_with_ollama(healed_reviews, model_info)
