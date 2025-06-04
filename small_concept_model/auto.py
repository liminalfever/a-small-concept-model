import json
import torch
from small_concept_model.model import SmallConceptModel
from small_concept_model.inverter import PreNet, Inverter, get_encoder, get_gpt2_decoder
from small_concept_model.pipeline import Pipeline

with open("small_concept_model/config.json", "r") as f:
    config = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_scm(model_id: str) -> SmallConceptModel:
    """Automatically builds SCM loading pre-trained weights."""

    assert model_id in config["model"].keys()

    model = SmallConceptModel(**config["model"][model_id]["configs"])
    model = model.to(device)
    model.load_state_dict(
        torch.load(
            config["model"][model_id]["pre_trained_weights"], map_location=device
        )
    )
    return model


def build_inverter(model_id: str) -> Inverter:
    """Automatically builds Inverter loading pre-trained weights."""

    assert model_id in config["inverter"].keys()

    prenet = PreNet(**config["inverter"][model_id]["configs"])
    prenet = prenet.to(device)
    prenet.load_state_dict(
        torch.load(
            config["inverter"][model_id]["pre_trained_weights"], map_location=device
        )
    )
    decoder, tokenizer = get_gpt2_decoder()

    prenet.eval()
    decoder.eval()

    return Inverter(prenet, decoder, tokenizer)


def build_pipeline(model_id: str, inverter_id: str) -> Pipeline:
    """Automatically builds a full SCM pipeline loading pre-trained weights."""

    assert model_id in config["model"].keys()
    assert inverter_id in config["inverter"].keys()

    encoder = get_encoder(config["inverter"][inverter_id]["encoder_id"])
    model = build_scm(model_id)
    inverter = build_inverter(inverter_id)

    return Pipeline(encoder, model, inverter)
