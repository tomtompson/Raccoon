import pytest

from raccoon.synthesizer.config import SynthesisConfig


def test_synthesis_config_accepts_dict_and_overrides():
    config = SynthesisConfig.from_inputs(
        {
            "output_dir": "out",
            "embedding_id": "emb",
            "query_model": "query",
            "query_validation_model": "qv",
            "judge_model": "judge",
            "qrel_validation_model": "dist",
            "dense_k": 3,
        },
        dense_k=7,
        embedding_kwargs={"model_kwargs": {"device": "cpu"}},
    )

    assert config.output_dir == "out"
    assert config.dense_k == 7
    assert config.embedding_kwargs == {"model_kwargs": {"device": "cpu"}}


def test_synthesis_config_requires_core_models():
    with pytest.raises(ValueError, match="query_model"):
        SynthesisConfig.from_inputs(
            output_dir="out",
            embedding_id="emb",
            query_validation_model="qv",
            judge_model="judge",
            qrel_validation_model="dist",
        )

