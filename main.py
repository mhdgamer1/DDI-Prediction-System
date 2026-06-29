

from typing import Optional, List
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ddi_engine import DDIEngine

# ---------------------------------------------------------------------- #
#  App + engine (engine loads ONCE at startup)                           #
# ---------------------------------------------------------------------- #
app = FastAPI(
    title="DDI Prediction API",
    description="Drug-drug interactions, severity, allergy cross-reactivity, "
                "and pregnancy safety . ",
    version="1.0.0",
)

# allow the frontend (any origin) to call this API.
# In production, replace ["*"] with your real frontend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load all models once when the server starts
engine = DDIEngine(data_dir="./")


# ---------------------------------------------------------------------- #
#  Health check                                                          #
# ---------------------------------------------------------------------- #
@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "DDI Prediction API",
        "endpoints": ["/resolve", "/interaction", "/allergy", "/pregnancy",
                      "/interaction/batch"],
        "docs": "/docs",
        "disclaimer": "Research/educational tool — not for clinical use.",
    }


# ---------------------------------------------------------------------- #
#  Name resolution                                                       #
# ---------------------------------------------------------------------- #
@app.get("/resolve")
def resolve(name: str = Query(..., description="Drug name (brand or generic)")):
    rxcui, source = engine.resolve_rxcui(name)
    if rxcui is None:
        raise HTTPException(status_code=404,
                            detail=f"Could not resolve '{name}'")
    return {"name": name, "rxcui": rxcui, "source": source}


# ---------------------------------------------------------------------- #
#  Interaction + severity + alternatives                                 #
# ---------------------------------------------------------------------- #
@app.get("/interaction")
def interaction(
    drug_a: str = Query(..., description="First drug name"),
    drug_b: str = Query(..., description="Second drug name"),
    alternatives: bool = Query(True, description="Include alternative suggestions"),
):
    result = engine.predict_interaction(drug_a, drug_b,
                                        include_alternatives=alternatives)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# ---------------------------------------------------------------------- #
#  Allergy cross-reactivity                                              #
# ---------------------------------------------------------------------- #
@app.get("/allergy")
def allergy(
    drug: str = Query(..., description="Drug the patient is allergic to"),
    max_results: int = Query(10, ge=1, le=50),
):
    result = engine.check_cross_reactivity(drug, max_results=max_results)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# ---------------------------------------------------------------------- #
#  Pregnancy safety (1 or 2 drugs)                                       #
# ---------------------------------------------------------------------- #
@app.get("/pregnancy")
def pregnancy(
    drug_a: str = Query(..., description="First drug name"),
    drug_b: Optional[str] = Query(None, description="Second drug (optional)"),
    live_api: bool = Query(True, description="Fetch live PLLR notes from openFDA"),
):
    result = engine.check_pregnancy(drug_a, drug_b, use_live_api=live_api)
    return result


# ---------------------------------------------------------------------- #
#  Batch interaction check (POST)                                        #
# ---------------------------------------------------------------------- #
class DrugPair(BaseModel):
    drug_a: str
    drug_b: str


class BatchRequest(BaseModel):
    pairs: List[DrugPair]


@app.post("/interaction/batch")
def interaction_batch(req: BatchRequest):
    """Check many drug pairs in one request. Useful for a full medication list."""
    results = []
    for pair in req.pairs:
        r = engine.predict_interaction(pair.drug_a, pair.drug_b)
        results.append(r)
    return {"count": len(results), "results": results}


# ---------------------------------------------------------------------- #
#  Full medication-list screen (POST) — checks every pair                #
# ---------------------------------------------------------------------- #
class MedListRequest(BaseModel):
    drugs: List[str]


@app.post("/screen")
def screen_medications(req: MedListRequest):
    """
    Given a list of drugs a patient takes, check ALL pairwise interactions
    and return only the ones that interact, sorted by severity.
    """
    from itertools import combinations
    sev_rank = {"Major": 3, "Moderate": 2, "Minor": 1, None: 0}
    findings = []
    for a, b in combinations(req.drugs, 2):
        r = engine.predict_interaction(a, b, include_alternatives=False)
        if "error" in r:
            continue
        if r["prediction"] == "Interaction likely":
            findings.append({
                "drug_a": a, "drug_b": b,
                "severity": r["severity"],
                "severity_confidence": r["severity_confidence"],
                "probability": r["interaction_probability"],
            })
    findings.sort(key=lambda x: sev_rank.get(x["severity"], 0), reverse=True)
    return {
        "drugs_checked": req.drugs,
        "pairs_checked": len(req.drugs) * (len(req.drugs) - 1) // 2,
        "interactions_found": len(findings),
        "findings": findings,
    }
