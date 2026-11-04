"""Human-readable labels for model features (display only; feature codes are always returned too)."""

from typing import Dict

FEATURE_LABELS: Dict[str, str] = {
    # engineered application-level
    "APP_EXT_SOURCES_MEAN": "External credit score (mean)",
    "APP_EXT_SOURCES_PROD": "External credit score (product)",
    "APP_EXT_SOURCES_MIN": "External credit score (min)",
    "APP_EXT_SOURCES_MAX": "External credit score (max)",
    "APP_EXT_SOURCES_RANGE": "External credit score (range)",
    "APP_PAYMENT_RATE": "Annuity-to-credit payment rate",
    "APP_GOODS_TO_CREDIT_RATIO": "Goods-to-credit ratio",
    "APP_CREDIT_TO_INCOME_RATIO": "Credit-to-income ratio",
    "APP_ANNUITY_TO_INCOME_PERCENT": "Annuity-to-income share",
    "APP_EMPLOYED_YEARS": "Years employed",
    "APP_AGE_YEARS": "Applicant age",
    "APP_INCOME_PER_PERSON": "Income per household member",
    "APP_DAYS_EMPLOYED_TO_BIRTH_RATIO": "Employment-to-age ratio",
    # cross-table derived
    "DERIVED_PAYMENT_DISCIPLINE_SCORE": "Payment discipline score",
    "DERIVED_TOTAL_PAID_TO_INCOME_RATIO": "Instalments paid to income",
    "DERIVED_TOTAL_DEBT_TO_INCOME_RATIO": "Total debt to income",
    # raw application
    "CODE_GENDER": "Applicant gender",
    "NAME_EDUCATION_TYPE": "Education level",
    "NAME_FAMILY_STATUS": "Family status",
    "NAME_INCOME_TYPE": "Income type",
    "AMT_ANNUITY": "Loan annuity",
    "AMT_CREDIT": "Loan amount",
    "AMT_GOODS_PRICE": "Goods price",
    "AMT_INCOME_TOTAL": "Annual income",
    "DAYS_BIRTH": "Age (days)",
    "DAYS_EMPLOYED": "Employment length (days)",
    "DAYS_ID_PUBLISH": "Days since ID issued",
    "OWN_CAR_AGE": "Car age",
    "EXT_SOURCE_1": "External score 1",
    "EXT_SOURCE_2": "External score 2",
    "EXT_SOURCE_3": "External score 3",
    # aggregates
    "POS_COMPLETION_RATE_MEAN": "POS loan completion rate",
    "INST_AMT_PAYMENT_SUM": "Total instalments paid",
    "BURO_DEBT_TO_CREDIT_RATIO": "Bureau debt-to-credit ratio",
    "BURO_ACTIVE_COUNT": "Active bureau credits",
    "PREV_REFUSAL_RATE": "Previous refusal rate",
    "PREV_CNT_PAYMENT_MEAN": "Previous loan term (mean)",
    "PREV_APPLICATION_TO_CREDIT_RATIO": "Previous applied-to-granted ratio",
    "CC_CNT_DRAWINGS_CURRENT_MEAN": "Card drawings per month",
}


def feature_label(feature: str) -> str:
    """Return the display label for a feature, or a readable fallback built from the code."""
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    return feature.replace("_", " ").title()
