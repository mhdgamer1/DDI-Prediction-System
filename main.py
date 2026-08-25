from typing import Optional, List
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import download_models
from ddi_engine import DDIEngine


# ============================================================
# Download DDI models from Hugging Face before loading engine
# ============================================================

download_models.download_models()




#  App + engine (engine loads ONCE at startup)                          

app = FastAPI(
    title="DDI Prediction API",
    description="Drug-drug interactions, severity, allergy cross-reactivity, "
                "and pregnancy safety . ",
    version="1.0.0",
)

# allow the frontend  to call this API.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load all models once when the server starts
engine = DDIEngine()

#  Health check                                                        

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


#  Name resolution                                                       

@app.get("/resolve")
def resolve(name: str = Query(..., description="Drug name (brand or generic)")):
    rxcui, source = engine.resolve_rxcui(name)
    if rxcui is None:
    
        suggestion, score = engine.suggest_name(name)
        if suggestion:
            return {
                "name": name,
                "rxcui": None,
                "resolved": False,
                "suggestion": suggestion,
                "suggestion_score": score,
                "message": f"Did you mean '{suggestion}'?",
            }
        raise HTTPException(status_code=404,
                            detail=f"Could not resolve '{name}'")
    return {"name": name, "rxcui": rxcui, "source": source, "resolved": True}



#  Interaction + severity + alternatives                                 

@app.get("/interaction")
def interaction(
    drug_a: str = Query(..., description="First drug name"),
    drug_b: str = Query(..., description="Second drug name"),
    alternatives: bool = Query(True, description="Include alternative suggestions"),
):
    result = engine.predict_interaction(drug_a, drug_b,
                                        include_alternatives=alternatives)
    if "error" in result:
   
        if "suggestion" in result:
            return result
        raise HTTPException(status_code=404, detail=result["error"])
    return result


class AllergyCheckRequest(BaseModel):
    medications: Optional[List[str]] = None   # للاستخدام الجديد
    allergies: Optional[List[str]] = None     # للاستخدام الجديد
    drug: Optional[str] = None                # للاستخدام القديم (توافقية خلفية)
    max_results: int = 10


#  Allergy cross-reactivity                                              

@app.post("/allergy")
def allergy(req: AllergyCheckRequest):
    """
    Two modes in one endpoint:
    - Single-drug mode (`drug`): returns everything structurally/pharmacologically
      similar to that one drug — the original lookup behaviour.
    - Prescription-check mode (`medications` + `allergies`): checks a list of
      prescribed drugs against a list of recorded allergies, flagging both
      exact-ingredient matches (e.g. Paracetamol prescribed against a Panadol
      allergy) and cross-reactive ones.
    """
    # ── Prescription-check mode ──
    if req.medications and req.allergies:
        all_direct = []
        all_cross = []

        for allergen in req.allergies:
            result = engine.check_cross_reactivity(
                allergen, check_against=req.medications, max_results=req.max_results
            )
            if "error" in result:
                continue
            for match in result.get("direct_matches", []):
                all_direct.append({**match, "allergen": allergen})
            for match in result.get("prescribed_matches", []):
                all_cross.append({**match, "allergen": allergen})

        return {
            "direct_matches": all_direct,
            "cross_reactive_matches": all_cross,
            "safe": len(all_direct) == 0 and len(all_cross) == 0,
        }

    # ── Single-drug mode (original behaviour) ──
    if req.drug:
        result = engine.check_cross_reactivity(req.drug, max_results=req.max_results)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result

    raise HTTPException(
        status_code=422,
        detail="Provide either 'drug' for a single lookup, or both "
               "'medications' and 'allergies' for a prescription check.",
    )

#  Pregnancy safety (1 or 2 drugs)                                      

@app.get("/pregnancy")
def pregnancy(
    drug_a: str = Query(..., description="First drug name"),
    drug_b: Optional[str] = Query(None, description="Second drug (optional)"),
    live_api: bool = Query(True, description="Fetch live PLLR notes from openFDA"),
):
    result = engine.check_pregnancy(drug_a, drug_b, use_live_api=live_api)
    return result

## Condition check 
class ConditionCheckRequest(BaseModel):
    medications: List[str]
    conditions: List[str]


@app.post("/condition-check")
def condition_check(req: ConditionCheckRequest):
    """Check prescribed drugs against the patient's chronic conditions."""
    result = engine.check_condition_contraindications(
        req.medications, req.conditions
    )
    return result

#  Batch interaction check (POST)                                       

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



#  Full medication-list screen (POST) — checks every pair                

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
