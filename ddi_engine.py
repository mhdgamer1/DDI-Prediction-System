
import re
import json
import joblib
import numpy as np
import pandas as pd
import networkx as nx
from rapidfuzz import process, fuzz

class DDIEngine:
    """Loads all models once and serves the four prediction functions."""

    ATC_PRIORITY = {
        'M': 1, 'N': 2, 'J': 3, 'C': 4, 'A': 5, 'B': 6, 'R': 7,
        'G': 8, 'L': 9, 'D': 10, 'H': 11, 'P': 12, 'S': 13, 'V': 14
    }
    PROPS = ['MolecularWeight', 'XLogP', 'TPSA', 'HBondDonorCount',
             'HBondAcceptorCount', 'RotatableBondCount', 'HeavyAtomCount']

    SALT_PATTERN = re.compile(
        r'\s+(sodium|potassium|calcium|magnesium|hydrochloride|sulfate|phosphate|'
        r'anhydrous|monohydrate|dihydrate|trihydrate|hemihydrate|pentahydrate|'
        r'acetate|citrate|tartrate|maleate|fumarate|succinate|besylate|mesylate|'
        r'tosylate|tromethamine|meglumine|benzathine|procaine|pivoxil|axetil|'
        r'proxetil|disodium|dipotassium|polistirex|tannate|camsyl|lactate|'
        r'napsylate|hydrate|anhyd|bitartrate|trometamol|terephthalate).*$',
        re.IGNORECASE
    )

    PREGNANCY_RULES = [
        (r'contraindicated.{0,80}pregnan|category\s*x', 'X',
         'Contraindicated in pregnancy — risk of fetal harm'),
        (r'teratogen|embryotoxic|fetal harm|fetal risk|neonatal.{0,50}death', 'D',
         'Evidence of fetal risk — use only if benefits outweigh risks'),
        (r'20 weeks|after 20|third trimester|last 3 months|oligohydramnios', 'D*',
         'Avoid in late pregnancy (after 20 weeks)'),
        (r'no adequate.{0,50}well.controlled|cross.{0,30}placenta|category\s*c|benefit.{0,50}outweigh', 'C',
         'Use with caution — no adequate human studies'),
        (r'category\s*b|no evidence.{0,50}harm|not reported a clear', 'B',
         'Relatively safe — no evidence of harm in animal studies'),
        (r'not sufficient to inform|insufficient data|limited.{0,50}data', 'N/A',
         'Insufficient data — consult physician'),
        (r'category\s*a', 'A',
         'Safe — controlled human studies show no risk'),
    ]

    PREG_RISK_RANK = {'X': 5, 'D': 4, 'D*': 3, 'C': 2, 'B': 1, 'A': 0,
                      'N/A': 1, 'Unknown': 1}
    PREG_ADVICE = {
        'X':  ' AVOID — one or both drugs are contraindicated in pregnancy',
        'D':  ' HIGH RISK — use only if no safe alternative exists',
        'D*': ' AVOID IN LATE PREGNANCY — especially after 20 weeks',
        'C':  ' USE WITH CAUTION — consult physician before use',
        'B':  ' RELATIVELY SAFE — but always confirm with physician',
        'A':  ' SAFE — generally considered safe in pregnancy',
    }


                        

    def __init__(self, data_dir="./"):
        d = data_dir.rstrip("/")

        
        self.rf       = joblib.load(f"{d}/ddi_final_model_v2.pkl")
        meta          = joblib.load(f"{d}/ddi_model_metadata_v2.pkl")
        support       = joblib.load(f"{d}/ddi_support_data_v2.pkl")
        self.feature_columns = meta["feature_columns"]

        self.ingredients       = support["ingredients"]
        self.bn_ingredient_map = support["bn_ingredient_map"]
        self.alt_names         = support["alt_names"]
        self.mol_props_full    = support["mol_props_full"]
        self.fp_matrix         = support["fp_matrix"]
        self.rxcui_to_idx      = support["rxcui_to_idx"]
        self.positive_set      = support["positive_set"]
        self.atc_lookup        = support["atc_lookup"]
        self.REGIONAL_SYNONYMS = support.get("regional_synonyms", {})
        self.PRIORITY_OVERRIDES = support.get("priority_overrides", {})
        self.rxcui_to_atc      = support.get("rxcui_to_atc", {})
        self.HIGH_RISK_ATC     = support.get("high_risk_atc", {})

        # severity models
        self.rf_major   = joblib.load(f"{d}/ddi_severity_major.pkl")
        self.rf_minor   = joblib.load(f"{d}/ddi_severity_minor.pkl")
        sev_config      = joblib.load(f"{d}/ddi_severity_config.pkl")
        self.MAJOR_THRESHOLD = sev_config["major_threshold"]
        self.MINOR_THRESHOLD = sev_config["minor_threshold"]
        self.train_medians   = pd.Series(sev_config["train_medians"])
        self.severity_feature_columns = sev_config.get(
            "severity_feature_columns", self.feature_columns)
        self.SEVERITY_OVERRIDES     = sev_config["overrides"]
        self.UNKNOWN_SEVERITY_PAIRS = sev_config.get("unknown_severity_pairs", set())

  
        allergy_config = joblib.load(f"{d}/ddi_allergy_config.pkl")
        self.TANIMOTO_THRESHOLD = allergy_config["tanimoto_threshold"]
        self.ATC_MIN_TANIMOTO   = allergy_config["atc_min_tanimoto"]
        self.PHARMACOPHORE_CLASSES = {
            k: set(v) for k, v in allergy_config["pharmacophore_classes"].items()
        }


        preg_data = joblib.load(f"{d}/ddi_pregnancy_data.pkl")
        self.PREGNANCY_OVERRIDES = preg_data["overrides"]
        self.pregnancy_db        = preg_data["pregnancy_db"]


        self.G = nx.Graph()
        self.G.add_edges_from(self.positive_set)
        self.fp_rxcuis = list(self.rxcui_to_idx.keys())
        self._m = self.mol_props_full.set_index("RXCUI")
        self._pure_ingredients = set(
            self.ingredients[self.ingredients["TTY"].isin(["IN", "PIN"])]["RXCUI"].tolist()
        )
        self._rxcui_to_name = self.ingredients.set_index("RXCUI")["ingredient_name"].to_dict()


                                               

    MIN_NAME_LENGTH = 3      
    ABBREV_MAX_LENGTH = 4     

    def _is_chemical_abbreviation(self, name_lower, str_value):
        """RxNorm stores chemical symbols as synonyms: "M"->methionine,
        "K"->potassium, "CET"->cephalothin. They are valid RxNorm rows, but a
        user typing 1-4 stray characters means a typo, not a drug. Real short
        ingredients (air, tin, urea) live in `ingredients` and are matched
        earlier, so they never reach this check."""
        return (len(name_lower) <= self.ABBREV_MAX_LENGTH
                and isinstance(str_value, str)
                and str_value.isupper())

    def resolve_rxcui(self, name):
        name_lower = name.lower().strip()
        if len(name_lower) < self.MIN_NAME_LENGTH:
            return None, None
        if name_lower in self.PRIORITY_OVERRIDES:
            return self.PRIORITY_OVERRIDES[name_lower], "priority_override"
        hit = self.ingredients[self.ingredients["name_lower"] == name_lower]
        if len(hit) > 0:
            return hit.iloc[0]["RXCUI"], "ingredient"
        bn = self.bn_ingredient_map.copy()
        bn["name_lower"] = bn["brand_name"].str.lower().str.strip()
        hit = bn[bn["name_lower"] == name_lower]
        if len(hit) > 0:
            ing = self.ingredients[
                self.ingredients["ingredient_name"] == hit.iloc[0]["ingredient_name"]]
            if len(ing) > 0:
                return ing.iloc[0]["RXCUI"], "brand_name"
        hit = self.alt_names[self.alt_names["name_lower"] == name_lower]
        if len(hit) > 0:
            if not self._is_chemical_abbreviation(name_lower, hit.iloc[0]["STR"]):
                return hit.iloc[0]["RXCUI"], "synonym"
        if name_lower in self.REGIONAL_SYNONYMS:
            return self.REGIONAL_SYNONYMS[name_lower], "regional_brand"
        return None, None

    def suggest_name(self, name, threshold=85):
        """Fuzzy-match a misspelled drug name against all known names.
        Returns (suggested_name, score) or (None, None). Does NOT auto-resolve —
        the caller should confirm the suggestion with the user before use."""
        name_lower = name.lower().strip()
        if len(name_lower) < self.MIN_NAME_LENGTH:
            return None, None
        if not hasattr(self, "_fuzzy_names"):
            names = set()
            names.update(self.ingredients["name_lower"].dropna().tolist())
     
            alt = self.alt_names.dropna(subset=["name_lower"])
            alt = alt[~alt.apply(
                lambda r: self._is_chemical_abbreviation(r["name_lower"], r["STR"]),
                axis=1)]
            names.update(alt["name_lower"].tolist())
            names.update(self.REGIONAL_SYNONYMS.keys())
            names.update(self.PRIORITY_OVERRIDES.keys())
            self._fuzzy_names = [n for n in names
                                 if isinstance(n, str)
                                 and len(n) >= self.MIN_NAME_LENGTH]
        match = process.extractOne(
            name_lower, self._fuzzy_names,
            scorer=fuzz.WRatio, score_cutoff=threshold)
        if match and match[0] != name_lower:
            return match[0], round(match[1], 1)
        return None, None




    def get_primary_atc_class(self, rxcui):
        classes = self.atc_lookup[self.atc_lookup["RXCUI"] == rxcui]["atc_class"].unique()
        if len(classes) == 0:
            return None
        sizes = {c: self.atc_lookup[self.atc_lookup["atc_class"] == c]["RXCUI"].nunique()
                 for c in classes}
        return min(classes, key=lambda c: (self.ATC_PRIORITY.get(c[0], 99), -sizes[c]))

    def get_alternatives(self, name_a, name_b, max_alternatives=5):
        rxcui_a, _ = self.resolve_rxcui(name_a)
        rxcui_b, _ = self.resolve_rxcui(name_b)
        if rxcui_a is None or rxcui_b is None:
            return {"error": "Could not resolve one or both drug names."}
        primary_class = self.get_primary_atc_class(rxcui_b)
        if primary_class is None:
            return {"message": f"No ATC class found for {name_b}."}
        same_class = [r for r in self.atc_lookup[
            self.atc_lookup["atc_class"] == primary_class]["RXCUI"].unique()
            if r != rxcui_b and r != rxcui_a]
        interacting_with_a = {(p[0] if p[1] == rxcui_a else p[1])
                              for p in self.positive_set if rxcui_a in p}
        safe = sorted(
            [r for r in same_class if r not in interacting_with_a],
            key=lambda r: self.G.degree(r) if self.G.has_node(r) else 0,
            reverse=True)
        candidates = [
            {"name": self._rxcui_to_name[r],
             "interaction_degree": self.G.degree(r) if self.G.has_node(r) else 0}
            for r in safe if r in self._rxcui_to_name][:max_alternatives]
        return {"drug_a": name_a, "drug_b": name_b, "atc_class": primary_class,
                "alternatives": candidates,
                "note": (f"Alternatives share ATC class {primary_class} with {name_b} "
                         f"and have no recorded interaction with {name_a} in DDInter. "
                         f"Always verify with a pharmacist.")}


    def _get_drug_risk_score(self, rxcui):
        rxcui = str(rxcui)
        atc_letter = self.rxcui_to_atc.get(rxcui, "")
        atc_score  = self.HIGH_RISK_ATC.get(atc_letter, 0)
        pharm_score = 0
        for cls, members in self.PHARMACOPHORE_CLASSES.items():
            if rxcui in members:
                pharm_score = 2 if cls in ["nsaid", "opioid", "statin"] else 1
                break
        return atc_score, pharm_score

    def _build_severity_features(self, feat, rxcui_a, rxcui_b):
        atc_a, pharm_a = self._get_drug_risk_score(rxcui_a)
        atc_b, pharm_b = self._get_drug_risk_score(rxcui_b)
        feat = dict(feat)
        feat["atc_risk_sum"]   = atc_a + atc_b
        feat["atc_risk_max"]   = max(atc_a, atc_b)
        feat["atc_risk_prod"]  = atc_a * atc_b
        feat["pharm_risk_sum"] = pharm_a + pharm_b
        feat["pharm_risk_max"] = max(pharm_a, pharm_b)
        feat["both_high_risk"] = 1 if (atc_a >= 2 and atc_b >= 2) else 0
        atc_la = self.rxcui_to_atc.get(str(rxcui_a), "")
        atc_lb = self.rxcui_to_atc.get(str(rxcui_b), "")
        feat["same_atc_class"]    = 1 if (atc_la and atc_la == atc_lb) else 0
        feat["blood_interaction"] = 1 if ("B" in [atc_la, atc_lb]) else 0
        return feat

    def _predict_severity(self, X, rxcui_a, rxcui_b):
        key = (min(rxcui_a, rxcui_b), max(rxcui_a, rxcui_b))
        if key in self.SEVERITY_OVERRIDES:
            return self.SEVERITY_OVERRIDES[key], "OVERRIDE", {
                "major_probability": None, "minor_probability": None,
                "note": "Severity from verified clinical override list"}
        base_feat = X.iloc[0].to_dict()
        enh_feat  = self._build_severity_features(base_feat, rxcui_a, rxcui_b)
        X_enh = pd.DataFrame([enh_feat])[self.severity_feature_columns].fillna(self.train_medians)
        str_key = (min(str(rxcui_a), str(rxcui_b)), max(str(rxcui_a), str(rxcui_b)))
        in_unknown = str_key in self.UNKNOWN_SEVERITY_PAIRS
        major_p = self.rf_major.predict_proba(X_enh)[0, 1]
        minor_p = self.rf_minor.predict_proba(X_enh)[0, 1]
        if major_p >= self.MAJOR_THRESHOLD:
            severity = "Major"; top_prob = major_p
        elif minor_p >= self.MINOR_THRESHOLD:
            severity = "Minor"; top_prob = minor_p
        else:
            severity = "Moderate"; top_prob = 1 - major_p
        if in_unknown:
            return severity, "UNCERTAIN", {
                "major_probability": round(float(major_p), 3),
                "minor_probability": round(float(minor_p), 3),
                "note": "DDInter records this interaction but does NOT classify its "
                        "severity. Model estimate only — verify clinically."}
        sev_conf = ("HIGH" if top_prob >= 0.70 else
                    "MEDIUM" if top_prob >= 0.55 else "LOW")
        return severity, sev_conf, {"major_probability": round(float(major_p), 3),
                                    "minor_probability": round(float(minor_p), 3)}


    def predict_interaction(self, name_a, name_b,
                            include_alternatives=True, max_alternatives=5):
        rxcui_a, src_a = self.resolve_rxcui(name_a)
        rxcui_b, src_b = self.resolve_rxcui(name_b)
        if rxcui_a is None:
            sugg, score = self.suggest_name(name_a)
            err = {"error": f"Could not find '{name_a}'"}
            if sugg:
                err["suggestion"] = sugg
                err["suggestion_score"] = score
                err["message"] = f"Did you mean '{sugg}'?"
            return err
        if rxcui_b is None:
            sugg, score = self.suggest_name(name_b)
            err = {"error": f"Could not find '{name_b}'"}
            if sugg:
                err["suggestion"] = sugg
                err["suggestion_score"] = score
                err["message"] = f"Did you mean '{sugg}'?"
            return err
        if rxcui_a == rxcui_b:
            return {"error": "Both names resolve to the same ingredient."}

        a, b = rxcui_a, rxcui_b
        override_key = (min(a, b), max(a, b))
        in_override  = override_key in self.SEVERITY_OVERRIDES
        in_a, in_b   = self.G.has_node(a), self.G.has_node(b)
        da = self.G.degree(a) if in_a else 0
        db = self.G.degree(b) if in_b else 0
        if in_a and in_b:
            na, nb  = set(self.G.neighbors(a)), set(self.G.neighbors(b))
            common  = len(na & nb); union = len(na | nb)
            jaccard = common / union if union else 0
            adamic  = sum(1 / np.log(self.G.degree(n))
                          for n in (na & nb) if self.G.degree(n) > 1)
        else:
            common = jaccard = adamic = 0

        feat = {"deg_a": da, "deg_b": db, "common_neighbors": common,
                "jaccard": jaccard, "adamic_adar": adamic}
        for p in self.PROPS:
            va = self._m.loc[a, p] if a in self._m.index else self._m[p].median()
            vb = self._m.loc[b, p] if b in self._m.index else self._m[p].median()
            feat[f"diff_{p}"] = abs(va - vb)
            feat[f"sum_{p}"]  = va + vb
        if a in self.rxcui_to_idx and b in self.rxcui_to_idx:
            va, vb = self.fp_matrix[self.rxcui_to_idx[a]], self.fp_matrix[self.rxcui_to_idx[b]]
            inter = np.logical_and(va, vb).sum(); uni = np.logical_or(va, vb).sum()
            feat["tanimoto"] = inter / uni if uni else 0
            feat["has_fingerprint"] = 1
        else:
            feat["tanimoto"] = 0
            feat["has_fingerprint"] = 0

        X = pd.DataFrame([feat])[self.feature_columns]
        if in_override:
            proba = 1.0; interacts = True
        else:
            proba = self.rf.predict_proba(X)[0, 1]; interacts = proba >= 0.5

        if interacts and in_a and in_b:
            severity, sev_conf, sev_probs = self._predict_severity(X, a, b)
        elif interacts and in_override:
            severity = self.SEVERITY_OVERRIDES[override_key]; sev_conf = "OVERRIDE"
            sev_probs = {"major_probability": None, "minor_probability": None,
                         "note": "Severity from verified clinical override list"}
        else:
            severity = sev_conf = sev_probs = None

        if interacts and include_alternatives:
            alt   = self.get_alternatives(name_a, name_b, max_alternatives)
            alts  = alt.get("alternatives", []); atc_c = alt.get("atc_class", None)
            alt_note = (f"Same ATC class as {name_b}, no recorded interaction "
                        f"with {name_a}. Always verify with a pharmacist.")
        else:
            alts = []; atc_c = None
            alt_note = "No interaction detected — no alternatives needed."

        if not in_a or not in_b:
            missing = ([name_a] if not in_a else []) + ([name_b] if not in_b else [])
            confidence = "NORMAL" if in_override else "LOW"
            warning = None if in_override else (
                f"{' and '.join(missing)} have no recorded interactions in DDInter. "
                f"This prediction is NOT reliable for clinical use.")
        else:
            confidence = "NORMAL"; warning = None

        result = {
            "drug_a": name_a, "resolved_as_a": src_a,
            "drug_b": name_b, "resolved_as_b": src_b,
            "interaction_probability": round(float(proba), 4),
            "prediction": "Interaction likely" if interacts else "No interaction expected",
            "severity": severity, "severity_confidence": sev_conf,
            "severity_confidence_explanation": {
                "OVERRIDE":  "Severity verified from DDInter clinical data — highest reliability",
                "HIGH":      "Model is >70% confident in this severity level",
                "MEDIUM":    "Model is 55-70% confident — use with caution",
                "LOW":       "Model confidence is below 55% — severity is uncertain",
                "UNCERTAIN": "DDInter confirms this interaction but does not classify its "
                             "severity. This label is a model estimate only — verify clinically",
            }.get(sev_conf, None),
            "severity_probabilities": sev_probs,
            "confidence": confidence,
            "known_in_training_data": (a, b) in self.positive_set or (b, a) in self.positive_set,
            "alternatives": {"atc_class": atc_c, "candidates": alts, "note": alt_note},
        }
        if warning:
            result["warning"] = warning
        return result


    def _get_base_name(self, drug_name):
        name = drug_name.lower().strip()
        name = name.split(",")[0].strip()
        name = self.SALT_PATTERN.sub("", name).strip()
        name = re.sub(r"\s*\(.*?\)", "", name).strip()
        return name

    def get_pharmacophore_class(self, rxcui):
        return [cls for cls, s in self.PHARMACOPHORE_CLASSES.items() if rxcui in s]

    def check_cross_reactivity(self, allergic_to, max_results=10,
                               tanimoto_threshold=None, atc_min_tanimoto=None):
        tanimoto_threshold = tanimoto_threshold or self.TANIMOTO_THRESHOLD
        atc_min_tanimoto   = atc_min_tanimoto   or self.ATC_MIN_TANIMOTO
        rxcui_a, src = self.resolve_rxcui(allergic_to)
        if rxcui_a is None:
            return {"error": f"Could not find '{allergic_to}' in database."}
        primary_class = self.get_primary_atc_class(rxcui_a)
        pharm_classes = self.get_pharmacophore_class(rxcui_a)
        same_atc_rxcuis = set(self.atc_lookup[
            self.atc_lookup["atc_class"] == primary_class]["RXCUI"].unique()
        ) if primary_class else set()
        same_pharm_rxcuis = set()
        for cls in pharm_classes:
            same_pharm_rxcuis.update(self.PHARMACOPHORE_CLASSES[cls])
        same_pharm_rxcuis.discard(rxcui_a)

        allergic_base = self._get_base_name(allergic_to)
        results = {}; seen_bases = {allergic_base}

        def try_add(rxcui_b, detected_by, tanimoto=None, risk="HIGH"):
            if rxcui_b in results or rxcui_b not in self._pure_ingredients:
                return
            name_vals = self.ingredients[
                self.ingredients["RXCUI"] == rxcui_b]["ingredient_name"].values
            if len(name_vals) == 0:
                return
            base = self._get_base_name(name_vals[0])
            if base in seen_bases:
                return
            seen_bases.add(base)
            results[rxcui_b] = {
                "name": base,
                "tanimoto": round(tanimoto, 4) if tanimoto is not None else None,
                "detected_by": detected_by, "risk": risk}

        # Layer 1: Tanimoto
        if rxcui_a in self.rxcui_to_idx:
            va = self.fp_matrix[self.rxcui_to_idx[rxcui_a]]
            inter = np.logical_and(self.fp_matrix, va).sum(axis=1)
            union = np.logical_or(self.fp_matrix, va).sum(axis=1)
            tan_scores = np.where(union > 0, inter / union, 0.0)
            for i, score in enumerate(tan_scores):
                rxcui_b = self.fp_rxcuis[i]
                if rxcui_b == rxcui_a or score < tanimoto_threshold:
                    continue
                if (rxcui_b not in same_atc_rxcuis and
                        rxcui_b not in same_pharm_rxcuis and score < 0.50):
                    continue
                try_add(rxcui_b, "structural_similarity", score,
                        "HIGH" if score >= 0.40 else "MODERATE")

        # Layer 2: Pharmacophore
        for rxcui_b in same_pharm_rxcuis:
            if rxcui_a in self.rxcui_to_idx and rxcui_b in self.rxcui_to_idx:
                va = self.fp_matrix[self.rxcui_to_idx[rxcui_a]]
                vb = self.fp_matrix[self.rxcui_to_idx[rxcui_b]]
                inter = np.logical_and(va, vb).sum(); union = np.logical_or(va, vb).sum()
                t = float(inter / union) if union > 0 else 0.0
            else:
                t = None
            if t is not None and t < atc_min_tanimoto:
                continue
            try_add(rxcui_b, "pharmacophore_class", t, "HIGH")

        # Layer 3: ATC
        for rxcui_b in same_atc_rxcuis:
            if rxcui_b == rxcui_a:
                continue
            if rxcui_a in self.rxcui_to_idx and rxcui_b in self.rxcui_to_idx:
                va = self.fp_matrix[self.rxcui_to_idx[rxcui_a]]
                vb = self.fp_matrix[self.rxcui_to_idx[rxcui_b]]
                inter = np.logical_and(va, vb).sum(); union = np.logical_or(va, vb).sum()
                t = float(inter / union) if union > 0 else 0.0
            else:
                t = None
            if t is None or t < atc_min_tanimoto:
                continue
            try_add(rxcui_b, "atc_class", t, "MODERATE")

        if not results:
            return {"allergic_to": allergic_to,
                    "message": "No cross-reactive drugs found.",
                    "atc_class": primary_class,
                    "pharmacophore_class": pharm_classes if pharm_classes else None}

        sorted_results = sorted(
            results.values(),
            key=lambda x: (0 if x["detected_by"] == "pharmacophore_class" else
                           1 if x["detected_by"] == "structural_similarity" else 2,
                           0 if x["tanimoto"] is not None else 1,
                           -(x["tanimoto"] or 0)))[:max_results]
        has_high = any(d["risk"] == "HIGH" for d in sorted_results)
        uncertainty_note = None if (pharm_classes or has_high) else (
            "No known pharmacophore cross-reactivity class for this drug. "
            "Results are based on structural similarity only and may NOT "
            "represent true clinical cross-reactivity risk. "
            "Consult a clinical pharmacist for guidance.")
        return {
            "allergic_to": allergic_to, "resolved_as": src,
            "atc_class": primary_class,
            "pharmacophore_class": pharm_classes if pharm_classes else None,
            "threshold_used": tanimoto_threshold,
            "cross_reactive_drugs": sorted_results,
            "uncertainty_note": uncertainty_note,
            "note": ("These drugs share structural similarity or pharmacological class "
                     "with the drug you are allergic to. Cross-reactivity risk varies. "
                     "Always consult a doctor or pharmacist before use.")}


    def _fetch_pregnancy_openfda(self, drug_name):
        import requests
        try:
            resp = requests.get(
                "https://api.fda.gov/drug/label.json",
                params={"search": f'openfda.generic_name:"{drug_name}"', "limit": 1},
                timeout=10)
            if resp.status_code != 200:
                return {"error": "API failed"}
            data = resp.json()
            if "results" not in data:
                return {"error": "No label found"}
            label = data["results"][0]
            fields = ["pregnancy", "pregnancy_or_breast_feeding",
                      "teratogenic_effects", "use_in_specific_populations"]
            return {"data": {
                f: " ".join(label[f]) if isinstance(label.get(f, []), list)
                else str(label.get(f, "")) for f in fields if f in label}}
        except Exception:
            return {"error": "API error"}

    def _extract_pregnancy_category(self, text, drug_name=None):
        if drug_name and drug_name.lower() in self.PREGNANCY_OVERRIDES:
            cat, warn = self.PREGNANCY_OVERRIDES[drug_name.lower()]
            return cat, warn, "clinical_override"
        text_lower = text.lower()
        for pattern, category, warning in self.PREGNANCY_RULES:
            if re.search(pattern, text_lower):
                return category, warning, "text_extraction"
        return "Unknown", "No pregnancy data found — consult physician", "not_found"

    def get_pregnancy_info(self, drug_name, use_live_api=True):
        rxcui, _ = self.resolve_rxcui(drug_name)
        ingredient = drug_name
        if rxcui:
            ing = self.ingredients[self.ingredients["RXCUI"] == rxcui]["ingredient_name"].values
            if len(ing) > 0:
                ingredient = ing[0]
        pllr_note = None
        if ingredient in self.pregnancy_db:
            data = self.pregnancy_db[ingredient]
            cat, warn, src = data["category"], data["warning"], data["source"]
            if use_live_api and src != "clinical_override":
                raw = self._fetch_pregnancy_openfda(ingredient)
                all_text = " ".join(raw.get("data", {}).values()) if "data" in raw else ""
                m = re.search(r"risk summary\s+(.{50,250}?)(?:\.|in animal|because|treatment)",
                              all_text, re.IGNORECASE)
                if m:
                    pllr_note = m.group(1).strip()
            return {"drug": drug_name, "ingredient": ingredient,
                    "category": cat, "warning": warn, "source": src,
                    "pllr_note": pllr_note}

        raw = self._fetch_pregnancy_openfda(ingredient) if use_live_api else {}
        all_text = " ".join(raw.get("data", {}).values()) if "data" in raw else ""
        cat, warn, src = self._extract_pregnancy_category(all_text, ingredient)
        m = re.search(r"risk summary\s+(.{50,250}?)(?:\.|in animal|because|treatment)",
                      all_text, re.IGNORECASE)
        if m:
            pllr_note = m.group(1).strip()
        return {"drug": drug_name, "ingredient": ingredient,
                "category": cat, "warning": warn, "source": src,
                "pllr_note": pllr_note}

    def check_pregnancy(self, name_a, name_b=None, use_live_api=True):
        ra = self.get_pregnancy_info(name_a, use_live_api)
        if name_b is None:
            return {"drug": name_a, "ingredient": ra["ingredient"],
                    "category": ra["category"], "warning": ra["warning"],
                    "pllr_note": ra.get("pllr_note"), "source": ra["source"],
                    "note": "Always consult a physician before taking any "
                            "medication during pregnancy."}
        rb = self.get_pregnancy_info(name_b, use_live_api)
        hi = ra["category"] if (self.PREG_RISK_RANK.get(ra["category"], 1) >=
                                self.PREG_RISK_RANK.get(rb["category"], 1)) else rb["category"]
        advice = self.PREG_ADVICE.get(hi, "⚪ UNKNOWN — consult physician")
        return {
            "drug_a": {"name": name_a, "ingredient": ra["ingredient"],
                       "category": ra["category"], "warning": ra["warning"],
                       "pllr_note": ra.get("pllr_note")},
            "drug_b": {"name": name_b, "ingredient": rb["ingredient"],
                       "category": rb["category"], "warning": rb["warning"],
                       "pllr_note": rb.get("pllr_note")},
            "overall_risk": hi, "overall_advice": advice,
            "note": "Always consult a physician before taking any medication during pregnancy."}



if __name__ == "__main__":
    engine = DDIEngine(data_dir="./")
    print("Engine loaded.\n")
    print(json.dumps(engine.predict_interaction("Warfarin", "Aspirin"), indent=2, ensure_ascii=False))
    print(json.dumps(engine.check_cross_reactivity("Penicillin G"), indent=2)[:500])
    print(json.dumps(engine.check_pregnancy("Warfarin", "Aspirin"), indent=2))
