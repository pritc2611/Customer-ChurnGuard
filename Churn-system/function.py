from pydantic import BaseModel , Field , field_validator
import pandas as pd
import shap
from pydantic import BaseModel, Field
import joblib
import shap
import pandas as pd
from pathlib import Path





def load_html_template(filename):
    template_path = Path("template") / filename
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()
    


class customerinput(BaseModel):
    gender:str = Field(...,description="customer gender")
    Subscription_type: str = Field(...,description="type of subscription")
    contract_length: str = Field(...,description="contract length. e.g: monthly ,qurtaly,annual")
    tenure: float = Field(...,ge=0,description="months with company")
    Age: int = Field(...,ge=18,le=100,description="customer age")
    Usage_Frequency: float = Field(...,ge=0,description="usage frequency. e.g: like days in month")
    Total_spend: float = Field(...,ge=0,description="total amount user spends so fare or recent month")

    @field_validator('gender')
    def validate_gender(cls,v):
        allowed = ["Male","Female"]
        if v not in allowed:
            raise ValueError(f'gender must be one of {allowed}')
        return v

    @field_validator('Subscription_type')
    def subcrip_val(cls,v):
        allowed = ["Standard","Premium","Basic"]
        if v not in allowed:
            raise ValueError(f'subscription type must be one of {allowed}')
        return v
    
    @field_validator('contract_length')
    def val_contract(cls,v):
        allowed = ['Annual', 'Monthly', 'Quarterly']
        if v not in allowed:
            raise ValueError(f'contract type must be one of {allowed}')
        return v

def generate_reasons(shap_val):
    shap_df = pd.DataFrame(
        {
            "feature": [
                "Gender_Female",
                "Gender_Male",
                "Subscription Type_Basic",
                "Subscription Type_Premium",
                "Subscription Type_Standard",
                "Contract Length_Annual",
                "Contract Length_Monthly",
                "Contract Length_Quarterly",
                "Age",
                "Tenure",
                "Usage Frequency",
                "Total Spend",
            ],
            "shap_value": shap_val.values[0, :, 1],  # class=1 (churn)
        }
    )

    # Sort by absolute importance
    shap_df = shap_df.reindex(
        shap_df.shap_value.abs().sort_values(ascending=False).index
    )

    # Filter weak effects
    shap_df = shap_df[shap_df.shap_value.abs() > 0.1].sort_values(
        by="shap_value", ascending=True
    )

    reasons = []
    for _, row in shap_df.iterrows():
        if row.shap_value > 0:
            reason = (
                f"{row.feature} of customer is increasing churn probability by "
                f"({row.shap_value:.2f})"
            )
        else:
            reason = (
                f"{row.feature} of customer is decreasing churn probability by "
                f"({row.shap_value:.2f})"
            )

        reasons.append(reason)

    return reasons

