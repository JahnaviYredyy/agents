"""Pydantic schemas for multi-agent debate system."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class ChallengeModel(BaseModel):
    claim: str = Field(description="The specific claim being challenged")
    reason: str = Field(description="Why the claim is questionable or unsupported")
    confidence: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Confidence level in the challenge",
    )
    suggested_correction: str = Field(
        description="A more accurate or balanced formulation"
    )


class DebateTurnModel(BaseModel):
    argument: str = Field(description="Your debate contribution")
    challenge: Optional[ChallengeModel] = Field(
        default=None,
        description="Optional challenge to an opposing claim",
    )


class CorrectionModel(BaseModel):
    decision: Literal["accepted", "revised", "defended"] = Field(
        description="Whether the claim is accepted, revised, or defended"
    )
    response: str = Field(description="Detailed explanation of the decision")
    corrected_claim: str = Field(
        default="",
        description="The modified/revised claim if revised or accepted, otherwise empty string",
    )


class CategoryScores(BaseModel):
    logic: int = Field(
        default=5, ge=1, le=10, description="Logical consistency and argumentation (1-10)"
    )
    evidence: int = Field(
        default=5, ge=1, le=10, description="Factual plausibility and grounding (1-10)"
    )
    rebuttal: int = Field(
        default=5, ge=1, le=10, description="Responsiveness and challenge defense (1-10)"
    )
    total: int = Field(default=0, description="Total score out of 30")

    def model_post_init(self, __context):
        if not self.total or self.total != (self.logic + self.evidence + self.rebuttal):
            self.total = self.logic + self.evidence + self.rebuttal


class EvaluationReportModel(BaseModel):
    winner: Literal["proponent", "opponent", "draw"] = Field(
        default="draw",
        description="The prevailing side in the debate"
    )
    proponent_scores: CategoryScores = Field(
        default_factory=CategoryScores,
        description="Scores for the proponent agent"
    )
    opponent_scores: CategoryScores = Field(
        default_factory=CategoryScores,
        description="Scores for the opponent agent"
    )
    key_concessions: List[str] = Field(
        default_factory=list,
        description="Key concessions or claim adjustments made during the debate",
    )
    unresolved_disagreements: List[str] = Field(
        default_factory=list,
        description="Core disagreements that remained unsettled",
    )
    report: str = Field(
        description="In-depth evaluation narrative analyzing both sides"
    )

