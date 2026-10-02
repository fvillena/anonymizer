"""NLP configuration and Presidio Analyzer factory."""

from typing import Optional

import torch
from presidio_analyzer import (
    AnalyzerEngine,
    EntityRecognizer,
    Pattern,
    PatternRecognizer,
    RecognizerResult,
)
from presidio_analyzer.nlp_engine import NlpArtifacts, NlpEngineProvider
from presidio_analyzer.predefined_recognizers import TransformersRecognizer
from presidio_analyzer.recognizer_registry import RecognizerRegistry
from transformers import pipeline

LANGUAGE = "es"

# Modelo principal para PERSON, ORGANIZATION y LOCATION.
SPACY_MODEL = "es_core_news_sm"
TRANSFORMERS_MODEL = "Babelscape/wikineural-multilingual-ner"

# Modelo secundario para fechas y horas en español.
MEDSPANER_MODEL = "medspaner/roberta-es-clinical-trials-temporal-ner"


class MedspanerTemporalRecognizer(EntityRecognizer):
    """Detect Spanish date and time expressions using MedSpaNER."""

    def __init__(
        self,
        supported_language: str = LANGUAGE,
        score_threshold: float = 0.50,
    ) -> None:
        """Load the MedSpaNER temporal NER model."""
        super().__init__(
            supported_entities=["DATE_TIME"],
            supported_language=supported_language,
            name="MedspanerTemporalRecognizer",
        )

        self.score_threshold = score_threshold

        # Usa GPU si PyTorch la detecta; si no, usa CPU.
        device = 0 if torch.cuda.is_available() else -1

        self.model = pipeline(
            task="token-classification",
            model=MEDSPANER_MODEL,
            tokenizer=MEDSPANER_MODEL,
            aggregation_strategy="simple",
            device=device,
        )

    def analyze(
        self,
        text: str,
        entities: list[str],
        nlp_artifacts: Optional[NlpArtifacts] = None,
    ) -> list[RecognizerResult]:
        """Return only DATE_TIME results inferred from Date and Time labels."""
        if entities and "DATE_TIME" not in entities:
            return []

        predictions = self.model(text)
        results = []

        for prediction in predictions:
            raw_label = (
                prediction.get("entity_group")
                or prediction.get("entity")
                or ""
            )

            # With aggregation_strategy="simple", this is normally Date or Time.
            model_label = (
                raw_label.upper()
                .replace("B-", "")
                .replace("I-", "")
            )

            if model_label != "DATE":
                continue

            score = float(prediction["score"])

            if score < self.score_threshold:
                continue

            results.append(
                RecognizerResult(
                    entity_type="DATE_TIME",
                    start=int(prediction["start"]),
                    end=int(prediction["end"]),
                    score=score,
                )
            )

        return results


def create_analyzer_engine() -> AnalyzerEngine:
    """Create a Spanish analyzer with custom recognizers."""
    nlp_configuration = {
        "nlp_engine_name": "transformers",
        "models": [
            {
                "lang_code": LANGUAGE,
                "model_name": {
                    "spacy": SPACY_MODEL,
                    "transformers": TRANSFORMERS_MODEL,
                },
            }
        ],
        "ner_model_configuration": {
            "labels_to_ignore": ["O"],
            "aggregation_strategy": "simple",
            "stride": 16,
            "alignment_mode": "expand",
            "model_to_presidio_entity_mapping": {
                "PER": "PERSON",
                "PERSON": "PERSON",
                "ORG": "ORGANIZATION",
                "ORGANIZATION": "ORGANIZATION",
                "LOC": "LOCATION",
                "LOCATION": "LOCATION",
            },
            "low_confidence_score_multiplier": 0.4,
            "low_score_entity_names": [],
        },
    }

    nlp_engine = NlpEngineProvider(
        nlp_configuration=nlp_configuration,
    ).create_engine()

    # Registry vacío: evita cargar todos los recognizers por defecto.
    registry = RecognizerRegistry(
        recognizers=[],
        supported_languages=[LANGUAGE],
    )

    # WikiNEuRal: personas, organizaciones y ubicaciones.
    registry.add_recognizer(
        TransformersRecognizer(
            supported_language=LANGUAGE,
            supported_entities=[
                "PERSON",
                "ORGANIZATION",
                "LOCATION",
            ],
        )
    )

    # Chilean RUT.
    rut_recognizer = PatternRecognizer(
        supported_entity="RUT",
        supported_language=LANGUAGE,
        name="ChileanRutRecognizer",
        patterns=[
            Pattern(
                name="Chilean RUT",
                regex=r"\b\d{1,2}(?:\.\d{3}){2}-[\dKk]\b|\b\d{7,8}-[\dKk]\b",
                score=0.85,
            )
        ],
        context=[
            "rut",
            "r.u.t",
            "r.u.t.",
            "rol único tributario",
            "rol unico tributario",
            "cédula",
            "cedula",
            "identificación",
            "identificacion",
        ],
    )
    registry.add_recognizer(rut_recognizer)

    # Chilean mobile and Santiago landline phone numbers.
    phone_recognizer = PatternRecognizer(
        supported_entity="PHONE_NUMBER",
        supported_language=LANGUAGE,
        name="ChileanPhoneRecognizer",
        patterns=[
            Pattern(
                name="Chilean mobile or landline phone number",
                regex=(
                    r"(?<!\d)"
                    r"(?:\+?56[\s.-]?)?"
                    r"(?:"
                    r"9[\s.-]?\d{4}[\s.-]?\d{4}"
                    r"|"
                    r"2[\s.-]?\d{4}[\s.-]?\d{4}"
                    r")"
                    r"(?!\d)"
                ),
                score=0.75,
            )
        ],
        context=[
            "teléfono",
            "telefono",
            "tel",
            "celular",
            "móvil",
            "movil",
            "fono",
            "llamar",
            "llámame",
            "llamame",
            "contacto",
            "whatsapp",
        ],
    )
    registry.add_recognizer(phone_recognizer)

    # MedSpaNER: solo expone Date y Time como DATE_TIME.
    registry.add_recognizer(
        MedspanerTemporalRecognizer(
            supported_language=LANGUAGE,
            score_threshold=0.50,
        )
    )

    return AnalyzerEngine(
        nlp_engine=nlp_engine,
        registry=registry,
        supported_languages=[LANGUAGE],
    )