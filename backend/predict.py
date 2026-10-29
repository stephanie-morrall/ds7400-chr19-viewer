"""Validates one cell's measurements and runs the saved chromosome classifier."""

import pandas as pd
import torch
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from training.features import features_for_cells
from training.prepare import feature_tensor

router = APIRouter()


async def validation_error_response(_request, error):
    """Returns validation details without echoing non-JSON input values."""
    # Non-finite input can be rejected before the prediction function is called.
    # Keeping only locations and messages also keeps that error response valid JSON.
    details = [{"loc": item["loc"], "msg": item["msg"], "type": item["type"]}
               for item in error.errors()]
    return JSONResponse(status_code=422, content={"detail": details})


class SpotIn(BaseModel):
    """Accepts only the identifiers and coordinates used by the feature function."""

    model_config = ConfigDict(extra="forbid")
    trace_id: int
    chrom_start: int
    x: FiniteFloat
    y: FiniteFloat
    z: FiniteFloat


class PredictBody(BaseModel):
    """One browser request supplies all measured spots for one selected cell."""

    model_config = ConfigDict(extra="forbid")
    spots: list[SpotIn] = Field(min_length=2, max_length=10000)


def spot_table(body):
    """Maps browser fields to the shared feature function's column names."""
    rows = [spot.model_dump() for spot in body.spots]
    table = pd.DataFrame(rows).rename(columns={
        "trace_id": "Trace_ID", "chrom_start": "Chrom_Start",
        "x": "X", "y": "Y", "z": "Z",
    })
    # The request represents one cell; its identifier is not a model input.
    return table.assign(Cell_ID=0)


def input_tensor(spots, predictor):
    """Filters and interpolates one cell using the saved training-locus mask."""
    metadata = predictor["metadata"]
    features, counts = features_for_cells(spots, metadata["cleaning"])
    if len(features) != 1:
        minimum = counts["minimum_observed_loci"]
        raise ValueError(
            "This cell does not pass the research quality filters: at most "
            f"4 traces per cell, with at least {minimum} measured retained loci per trace."
        )
    if list(features.columns) != metadata["feature_names"]:
        raise RuntimeError("Feature order differs from the saved model.")
    # Distances retain their original physical scale without fitting a transform.
    return feature_tensor(features)


def predict_cell(spots, predictor):
    """Returns the highest-scoring supported type and its model identity."""
    inputs = input_tensor(spots, predictor)
    with torch.inference_mode():
        scores = predictor["model"](inputs).softmax(dim=1)[0]
    if not torch.isfinite(scores).all():
        raise RuntimeError("The model returned a non-finite score.")
    index = int(scores.argmax())
    metadata = predictor["metadata"]
    return {
        "predicted_type": metadata["class_names"][index],
        "model_score": float(scores[index]),
        "model_version": metadata["model_version"],
        "class_names": metadata["class_names"],
    }


@router.post("/api/predict")
def predict(body: PredictBody, request: Request):
    """Serves the same saved predictor used for the frozen model evaluation."""
    try:
        return predict_cell(spot_table(body), request.app.state.predictor)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
