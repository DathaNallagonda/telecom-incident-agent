"""Pydantic schemas for agent outputs.

Ids are checked against the live network when a `network` is passed in the
validation context: `TriageResult.model_validate(data, context={"network": state})`.
Without a context (e.g. while the LLM library parses raw output) only the
shape is checked.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

FaultType = Literal["power", "hardware", "link", "unknown"]
Priority = Literal["P1", "P2", "P3"]


def _network(info: ValidationInfo):
    return (info.context or {}).get("network")


class NodeFault(BaseModel):
    node_id: str = Field(description="Node id, e.g. T3")
    fault_type: FaultType = Field(description="power, hardware, link or unknown")


class LinkRef(BaseModel):
    a: str = Field(description="Node id at one end")
    b: str = Field(description="Node id at the other end")


class TriageResult(BaseModel):
    faulty_nodes: list[str] = Field(description="Ids of nodes that are faulty")
    fault_types: list[NodeFault] = Field(description="One entry per faulty node")
    affected_sectors: list[str] = Field(description="Ids of sectors that lost service")
    cut_links: list[LinkRef] = Field(description="Links that are down")
    priority: Priority = Field(description="P1 = many customers or no backup, P3 = minor")
    confidence: float = Field(description="0 to 1")
    reasoning: str = Field(description="One or two short sentences")

    @field_validator("confidence")
    @classmethod
    def _confidence_range(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("confidence must be between 0 and 1")
        return v

    @field_validator("faulty_nodes")
    @classmethod
    def _nodes_exist(cls, v: list[str], info: ValidationInfo) -> list[str]:
        net = _network(info)
        if net is not None:
            bad = [n for n in v if n not in net.nodes]
            if bad:
                raise ValueError(f"unknown node ids {bad}; known: {sorted(net.nodes)}")
        return v

    @field_validator("affected_sectors")
    @classmethod
    def _sectors_exist(cls, v: list[str], info: ValidationInfo) -> list[str]:
        net = _network(info)
        if net is not None:
            bad = [s for s in v if s not in net.sectors]
            if bad:
                raise ValueError(f"unknown sector ids {bad}; known: {sorted(net.sectors)}")
        return v

    @field_validator("cut_links")
    @classmethod
    def _links_exist(cls, v: list[LinkRef], info: ValidationInfo) -> list[LinkRef]:
        net = _network(info)
        if net is not None:
            bad = [f"{lk.a}-{lk.b}" for lk in v if net.link(lk.a, lk.b) is None]
            if bad:
                raise ValueError(f"unknown links {bad}")
        return v

    @model_validator(mode="after")
    def _fault_types_match_nodes(self, info: ValidationInfo) -> TriageResult:
        typed = [f.node_id for f in self.fault_types]
        missing = [n for n in self.faulty_nodes if n not in typed]
        extra = [n for n in typed if n not in self.faulty_nodes]
        if missing:
            raise ValueError(f"fault_types has no entry for faulty nodes {missing}")
        if extra:
            raise ValueError(f"fault_types lists nodes not in faulty_nodes: {extra}")
        return self

    def fault_of(self, node_id: str) -> FaultType | None:
        return next((f.fault_type for f in self.fault_types if f.node_id == node_id), None)


class PlanStep(BaseModel):
    action: str = Field(description="Action name, e.g. restart_node")
    args: list[str] = Field(description="Arguments in the order the action lists them")


class PlanDraft(BaseModel):
    steps: list[PlanStep] = Field(description="The plan, in execution order")
