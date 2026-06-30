from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field


class Status(str, Enum):
    A11 = "A11"  # < 0 DM
    A12 = "A12"  # 0 <= x < 200 DM
    A13 = "A13"  # >= 200 DM
    A14 = "A14"  # no checking account


class CreditHistory(str, Enum):
    A30 = "A30"  # no credits taken / all paid back duly
    A31 = "A31"  # all credits at this bank paid back duly
    A32 = "A32"  # existing credits paid back duly till now
    A33 = "A33"  # delay in paying off in the past
    A34 = "A34"  # critical account / other credits existing


class Purpose(str, Enum):
    A40 = "A40"  # car (new)
    A41 = "A41"  # car (used)
    A42 = "A42"  # furniture/equipment
    A43 = "A43"  # radio/television
    A44 = "A44"  # domestic appliances
    A45 = "A45"  # repairs
    A46 = "A46"  # education
    A48 = "A48"  # retraining
    A49 = "A49"  # business
    A410 = "A410"  # others


class Savings(str, Enum):
    A61 = "A61"  # < 100 DM
    A62 = "A62"  # 100 <= x < 500 DM
    A63 = "A63"  # 500 <= x < 1000 DM
    A64 = "A64"  # >= 1000 DM
    A65 = "A65"  # unknown / no savings account


class EmploymentDuration(str, Enum):
    A71 = "A71"  # unemployed
    A72 = "A72"  # < 1 year
    A73 = "A73"  # 1 <= x < 4 years
    A74 = "A74"  # 4 <= x < 7 years
    A75 = "A75"  # >= 7 years


class PersonalStatusSex(str, Enum):
    A91 = "A91"  # male: divorced/separated
    A92 = "A92"  # female: divorced/separated/married
    A93 = "A93"  # male: single
    A94 = "A94"  # male: married/widowed


class OtherDebtors(str, Enum):
    A101 = "A101"  # none
    A102 = "A102"  # co-applicant
    A103 = "A103"  # guarantor


class Property(str, Enum):
    A121 = "A121"  # real estate
    A122 = "A122"  # building society savings / life insurance
    A123 = "A123"  # car or other
    A124 = "A124"  # unknown / no property


class OtherInstallmentPlans(str, Enum):
    A141 = "A141"  # bank
    A142 = "A142"  # stores
    A143 = "A143"  # none


class Housing(str, Enum):
    A151 = "A151"  # rent
    A152 = "A152"  # own
    A153 = "A153"  # for free


class Job(str, Enum):
    A171 = "A171"  # unemployed / unskilled non-resident
    A172 = "A172"  # unskilled resident
    A173 = "A173"  # skilled employee / official
    A174 = "A174"  # management / self-employed / highly qualified


class Telephone(str, Enum):
    A191 = "A191"  # none
    A192 = "A192"  # yes, registered under customer name


class ForeignWorker(str, Enum):
    A201 = "A201"  # yes
    A202 = "A202"  # no


class ApplicantFeatures(BaseModel):
    Status: Status
    Duration: Annotated[int, Field(gt=0, description="Credit duration in months")]
    CreditHistory: CreditHistory
    Purpose: Purpose
    CreditAmount: Annotated[int, Field(gt=0, description="Credit amount in DM")]
    Savings: Savings
    EmploymentDuration: EmploymentDuration
    InstallmentRate: Annotated[int, Field(ge=1, le=4, description="Installment rate as % of disposable income")]
    PersonalStatusSex: PersonalStatusSex
    OtherDebtors: OtherDebtors
    ResidenceDuration: Annotated[int, Field(ge=1, le=4, description="Years at present residence")]
    Property: Property
    Age: Annotated[int, Field(gt=0, description="Age in years")]
    OtherInstallmentPlans: OtherInstallmentPlans
    Housing: Housing
    ExistingCredits: Annotated[int, Field(ge=1, le=4, description="Number of existing credits at this bank")]
    Job: Job
    NumberOfDependents: Annotated[int, Field(ge=1, le=2, description="Number of dependents")]
    Telephone: Telephone
    ForeignWorker: ForeignWorker

    model_config = {
        "json_schema_extra": {
            "example": {
                "Status": "A11",
                "Duration": 24,
                "CreditHistory": "A32",
                "Purpose": "A43",
                "CreditAmount": 4000,
                "Savings": "A61",
                "EmploymentDuration": "A73",
                "InstallmentRate": 2,
                "PersonalStatusSex": "A93",
                "OtherDebtors": "A101",
                "ResidenceDuration": 2,
                "Property": "A121",
                "Age": 35,
                "OtherInstallmentPlans": "A143",
                "Housing": "A152",
                "ExistingCredits": 1,
                "Job": "A173",
                "NumberOfDependents": 1,
                "Telephone": "A191",
                "ForeignWorker": "A201",
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: int = Field(description="0 = low risk, 1 = high risk")
    probability_high_risk: float
    model_used: str


class ExplainResponse(PredictionResponse):
    top_features: list[dict] = Field(
        description="Top SHAP feature contributions for this prediction"
    )
