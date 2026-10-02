"""FastAPI server for Presidio Analyzer and Anonymizer."""

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from presidio_analyzer import AnalyzerEngine, AnalyzerRequest, BatchAnalyzerEngine
from presidio_anonymizer import AnonymizerEngine, DeanonymizeEngine
from presidio_anonymizer.entities import InvalidParamError
from presidio_anonymizer.services.app_entities_convertor import AppEntitiesConvertor
from anonymizers import register_anonymizers

from analyzers import LANGUAGE, SPACY_MODEL, TRANSFORMERS_MODEL, create_analyzer_engine

DEFAULT_PORT = 3000
DEFAULT_BATCH_SIZE = 500
DEFAULT_N_PROCESS = 1

logger = logging.getLogger("presidio-api")


def remove_internal_attributes(results: list[Any]) -> None:
    """Remove internal Presidio metadata before returning results."""
    for result in results:
        if hasattr(result, "recognition_metadata"):
            delattr(result, "recognition_metadata")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Presidio once when the API starts."""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logger.info("Starting Presidio Analyzer")
    logger.info("Language: %s", LANGUAGE)
    logger.info("spaCy model: %s", SPACY_MODEL)
    logger.info("Transformers model: %s", TRANSFORMERS_MODEL)

    analyzer_engine = create_analyzer_engine()

    app.state.analyzer_engine = analyzer_engine
    app.state.batch_engine = BatchAnalyzerEngine(analyzer_engine)
    anonymizer_engine = AnonymizerEngine()
    register_anonymizers(anonymizer_engine)

    app.state.anonymizer_engine = anonymizer_engine
    app.state.deanonymizer_engine = DeanonymizeEngine()

    logger.info("Presidio API started")

    yield

    logger.info("Presidio API stopped")


app = FastAPI(
    title="Presidio Analyzer and Anonymizer API",
    description="Spanish PII analysis with Transformers and Presidio anonymization.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    """Use a consistent JSON error format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )


@app.get("/health", response_class=PlainTextResponse)
def health() -> str:
    """Return service health."""
    return "Presidio Analyzer and Anonymizer service is up"


@app.post("/analyze")
async def analyze(request: Request) -> JSONResponse:
    """Analyze one text or a list of texts."""
    try:
        content = await request.json()
        analyzer_request = AnalyzerRequest(content)

        if not analyzer_request.text:
            raise HTTPException(status_code=400, detail="No text provided")

        if analyzer_request.language != LANGUAGE:
            raise HTTPException(
                status_code=400,
                detail=f"Only '{LANGUAGE}' is supported",
            )

        analyzer_engine: AnalyzerEngine = request.app.state.analyzer_engine
        batch_engine: BatchAnalyzerEngine = request.app.state.batch_engine

        is_batch = isinstance(analyzer_request.text, list)
        texts = analyzer_request.text if is_batch else [analyzer_request.text]

        if not all(isinstance(text, str) and text for text in texts):
            raise HTTPException(
                status_code=400,
                detail="Each text must be a non-empty string",
            )

        batch_size = min(
            len(texts),
            int(os.getenv("BATCH_SIZE", DEFAULT_BATCH_SIZE)),
        )

        n_process = min(
            len(texts),
            int(os.getenv("N_PROCESS", DEFAULT_N_PROCESS)),
        )

        iterator = batch_engine.analyze_iterator(
            texts=texts,
            language=analyzer_request.language,
            batch_size=batch_size,
            n_process=n_process,
            entities=analyzer_request.entities,
            score_threshold=analyzer_request.score_threshold,
            correlation_id=analyzer_request.correlation_id,
            return_decision_process=analyzer_request.return_decision_process,
            ad_hoc_recognizers=analyzer_request.ad_hoc_recognizers,
            context=analyzer_request.context,
            allow_list=analyzer_request.allow_list,
            allow_list_match=analyzer_request.allow_list_match,
            regex_flags=analyzer_request.regex_flags,
        )

        results = []

        for text_results in iterator:
            remove_internal_attributes(text_results)
            results.append(text_results)

        response_data = results if is_batch else results[0]

        return JSONResponse(
            content=json.loads(
                json.dumps(
                    response_data,
                    default=lambda item: item.to_dict(),
                    sort_keys=True,
                )
            )
        )

    except HTTPException:
        raise

    except TypeError as exc:
        logger.exception("Invalid /analyze request")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid analyze request: {exc}",
        ) from exc

    except Exception as exc:
        logger.exception("PII analysis failed")
        raise HTTPException(
            status_code=500,
            detail=f"PII analysis failed: {exc}",
        ) from exc


@app.post("/anonymize")
async def anonymize(request: Request) -> JSONResponse:
    """Anonymize text with results returned by POST /analyze."""
    try:
        content = await request.json()

        if not content:
            raise HTTPException(status_code=400, detail="Invalid request JSON")

        if "text" not in content:
            raise HTTPException(status_code=400, detail="No text provided")

        if "analyzer_results" not in content:
            raise HTTPException(
                status_code=400,
                detail="No analyzer_results provided",
            )

        operators = AppEntitiesConvertor.operators_config_from_json(
            content.get("anonymizers")
        )

        if AppEntitiesConvertor.check_custom_operator(operators):
            raise HTTPException(
                status_code=400,
                detail="Custom anonymizer operators are not supported",
            )

        analyzer_results = AppEntitiesConvertor.analyzer_results_from_json(
            content["analyzer_results"]
        )

        anonymizer: AnonymizerEngine = request.app.state.anonymizer_engine

        result = anonymizer.anonymize(
            text=content["text"],
            analyzer_results=analyzer_results,
            operators=operators,
        )

        return JSONResponse(content=json.loads(result.to_json()))

    except HTTPException:
        raise

    except InvalidParamError as exc:
        raise HTTPException(
            status_code=422,
            detail=exc.err_msg,
        ) from exc

    except Exception as exc:
        logger.exception("Anonymization failed")
        raise HTTPException(
            status_code=500,
            detail=f"Anonymization failed: {exc}",
        ) from exc


@app.post("/deanonymize")
async def deanonymize(request: Request) -> JSONResponse:
    """Deanonymize data created with Presidio's encrypt operator."""
    try:
        content = await request.json()

        if not content:
            raise HTTPException(status_code=400, detail="Invalid request JSON")

        if "text" not in content:
            raise HTTPException(status_code=400, detail="No text provided")

        entities = AppEntitiesConvertor.deanonymize_entities_from_json(content)

        operators = AppEntitiesConvertor.operators_config_from_json(
            content.get("deanonymizers")
        )

        deanonymizer: DeanonymizeEngine = request.app.state.deanonymizer_engine

        result = deanonymizer.deanonymize(
            text=content["text"],
            entities=entities,
            operators=operators,
        )

        return JSONResponse(content=json.loads(result.to_json()))

    except HTTPException:
        raise

    except InvalidParamError as exc:
        raise HTTPException(
            status_code=422,
            detail=exc.err_msg,
        ) from exc

    except Exception as exc:
        logger.exception("Deanonymization failed")
        raise HTTPException(
            status_code=500,
            detail=f"Deanonymization failed: {exc}",
        ) from exc


@app.post("/analyze+anonymize")
async def analyze_and_anonymize(request: Request) -> JSONResponse:
    """
    Analyze and anonymize a single text in one request.

    Example:
    {
      "text": "Fabián Villena trabaja en Microsoft Chile.",
      "language": "es",
      "anonymizers": {
        "PERSON": {
          "type": "replace",
          "new_value": "<PERSONA>"
        },
        "ORGANIZATION": {
          "type": "replace",
          "new_value": "<ORGANIZACION>"
        },
        "LOCATION": {
          "type": "replace",
          "new_value": "<UBICACION>"
        },
        "DEFAULT": {
          "type": "replace",
          "new_value": "<PII>"
        }
      }
    }
    """
    try:
        content = await request.json()
        analyzer_request = AnalyzerRequest(content)

        if not analyzer_request.text:
            raise HTTPException(
                status_code=400,
                detail="No text provided",
            )

        if not isinstance(analyzer_request.text, str):
            raise HTTPException(
                status_code=400,
                detail=(
                    "The analyze-and-anonymize endpoint accepts "
                    "only one text string"
                ),
            )

        if analyzer_request.language != LANGUAGE:
            raise HTTPException(
                status_code=400,
                detail=f"Only '{LANGUAGE}' is supported",
            )

        analyzer_engine: AnalyzerEngine = request.app.state.analyzer_engine
        anonymizer: AnonymizerEngine = request.app.state.anonymizer_engine

        analyzer_results = analyzer_engine.analyze(
            text=analyzer_request.text,
            language=analyzer_request.language,
            entities=analyzer_request.entities,
            score_threshold=analyzer_request.score_threshold,
            correlation_id=analyzer_request.correlation_id,
            return_decision_process=analyzer_request.return_decision_process,
            ad_hoc_recognizers=analyzer_request.ad_hoc_recognizers,
            context=analyzer_request.context,
            allow_list=analyzer_request.allow_list,
            allow_list_match=analyzer_request.allow_list_match,
            regex_flags=analyzer_request.regex_flags,
        )

        anonymizers_config = AppEntitiesConvertor.operators_config_from_json(
            content.get("anonymizers")
        )

        if AppEntitiesConvertor.check_custom_operator(anonymizers_config):
            raise HTTPException(
                status_code=400,
                detail="Custom anonymizer operators are not supported",
            )

        anonymized_result = anonymizer.anonymize(
            text=analyzer_request.text,
            analyzer_results=analyzer_results,
            operators=anonymizers_config,
        )

        remove_internal_attributes(analyzer_results)

        return JSONResponse(
            content={
                "analyzer_results": json.loads(
                    json.dumps(
                        analyzer_results,
                        default=lambda item: item.to_dict(),
                        sort_keys=True,
                    )
                ),
                "anonymized_result": json.loads(
                    anonymized_result.to_json()
                ),
            }
        )

    except HTTPException:
        raise

    except InvalidParamError as exc:
        raise HTTPException(
            status_code=422,
            detail=exc.err_msg,
        ) from exc

    except TypeError as exc:
        logger.exception("Invalid /analyze-and-anonymize request")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request: {exc}",
        ) from exc

    except Exception as exc:
        logger.exception("Analyze and anonymize operation failed")
        raise HTTPException(
            status_code=500,
            detail=f"Analyze and anonymize operation failed: {exc}",
        ) from exc

@app.get("/recognizers")
def recognizers(
    request: Request,
    language: str = Query(default=LANGUAGE),
) -> list[str]:
    """Return recognizers available for Spanish."""
    try:
        analyzer_engine: AnalyzerEngine = request.app.state.analyzer_engine

        return [
            recognizer.name
            for recognizer in analyzer_engine.get_recognizers(language)
        ]

    except Exception as exc:
        logger.exception("Could not retrieve recognizers")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/supportedentities")
def supported_entities(
    request: Request,
    language: str = Query(default=LANGUAGE),
) -> list[str]:
    """Return entity types supported by the current NLP engine."""
    try:
        analyzer_engine: AnalyzerEngine = request.app.state.analyzer_engine

        return analyzer_engine.get_supported_entities(language)

    except Exception as exc:
        logger.exception("Could not retrieve supported entities")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/anonymizers")
def anonymizers(request: Request) -> Any:
    """Return available Presidio anonymization operators."""
    anonymizer: AnonymizerEngine = request.app.state.anonymizer_engine
    return anonymizer.get_anonymizers()


@app.get("/deanonymizers")
def deanonymizers(request: Request) -> Any:
    """Return available Presidio deanonymization operators."""
    deanonymizer: DeanonymizeEngine = request.app.state.deanonymizer_engine
    return deanonymizer.get_deanonymizers()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", DEFAULT_PORT)),
    )