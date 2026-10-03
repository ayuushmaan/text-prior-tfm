"""Semantic grounding schemas.

A synthetic prior (TabICL-v2 / nanoprior) produces anonymous numeric columns
and integer categorical codes. A schema assigns each column a real-world
meaning (name, unit, plausible range) and each categorical code a word, so
generated text is *coherent by construction*: every mention in the text is a
deterministic function of that row's X values.
"""

DOMAINS = {
    "clinic": {
        "entity": "patient",
        "num_features": [
            ("age", "years old", 18, 90),
            ("blood_pressure", "mmHg", 90, 180),
            ("cholesterol", "mg/dL", 120, 300),
        ],
        "cat_features": {
            "smoker": ["non-smoker", "smoker"],
            "gender": ["female", "male"],
        },
        "templates": [
            "{entity} aged {age} ({gender}, {smoker}) presented with BP {blood_pressure} and cholesterol {cholesterol}.",
            "{age}-year-old {gender} {entity} ({smoker}); vitals: BP {blood_pressure} mmHg, cholesterol {cholesterol} mg/dL.",
        ],
    },
    "loan": {
        "entity": "applicant",
        "num_features": [
            ("income", "k$/yr", 20, 200),
            ("debt_ratio", "", 0.0, 1.0),
            ("tenure", "years", 0, 30),
        ],
        "cat_features": {
            "employed": ["unemployed", "employed"],
            "region": ["north", "south", "east", "west"],
        },
        "templates": [
            "{entity} with income {income}k, debt ratio {debt_ratio}, tenure {tenure}y ({employed}, {region} region).",
            "{employed} {entity} from {region}: income {income}k$/yr, debt {debt_ratio}, {tenure} years tenure.",
        ],
    },
}


def ground_columns(n_num, n_cat, domain_name="clinic"):
    """Assign schema names to the first n_num numeric and n_cat categorical columns.

    Extra columns beyond the schema fall back to generic ``num_N`` / ``cat_N``
    names so any prior shape works (all-numeric TabICL tables included).
    """
    if domain_name not in DOMAINS:
        raise ValueError(f"unknown domain {domain_name!r}, choose from {sorted(DOMAINS)}")
    d = DOMAINS[domain_name]
    num_names = [f[0] for f in d["num_features"][: max(1, n_num)]]
    while len(num_names) < n_num:
        num_names.append(f"num_{len(num_names)}")
    cat_names = list(d["cat_features"].keys())[:n_cat]
    while len(cat_names) < n_cat:
        cat_names.append(f"cat_{len(cat_names)}")
    return d, num_names[:n_num], cat_names[:n_cat]
