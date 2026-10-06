"""
Domain models and schema definitions for the TaskFlow workspace.
"""

from decimal import Decimal
from datetime import date
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator

import re

def parse_money_to_minor(amount_input: str | int | float) -> int:
    """
    Safely converts string or numeric money (e.g. '84,500', '84500.00', '84,500 INR', 84500)
    to integer minor units (paise/cents) using Python's Decimal.
    Avoids floating-point precision errors.
    """
    value = str(amount_input).strip()
    value = re.sub(r"^(?:INR|USD|EUR|GBP|₹|\$|€|£)\s*|\s*(?:INR|USD|EUR|GBP)$", "", value, flags=re.I).strip()
    if not re.fullmatch(r"(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d{1,2})?", value):
        raise ValueError("Amount must be a nonnegative monetary value with at most two decimals")
    return int(Decimal(value.replace(",", "")) * 100)

def format_minor_to_money(minor: int, currency: str = "INR") -> str:
    """Formats integer minor units to human-readable string with 2 decimals."""
    dec = Decimal(minor) / Decimal("100")
    symbol = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}.get(currency.upper(), currency.upper() + " ")
    return f"{symbol}{dec:,.2f}"

class InvoiceCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    company: str = Field(..., min_length=1)
    invoice_number: str = Field(..., min_length=1)
    amount: str | int | float
    currency: str = Field(default="INR", pattern=r"^(INR|USD|EUR|GBP)$")
    due_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    status: str = Field(default="Pending", pattern=r"^(Pending|Paid|Overdue)$")
    source_reference: Optional[str] = None

    @field_validator("due_date")
    @classmethod
    def valid_date(cls, value):
        date.fromisoformat(value)
        return value

    @field_validator("amount")
    @classmethod
    def valid_amount(cls, value):
        parse_money_to_minor(value)
        return value


class InvoiceRecord(BaseModel):
    id: int
    company: str
    invoice_number: str
    amount_minor: int
    amount_formatted: str
    currency: str
    due_date: str
    status: str
    created_at: str
    source_reference: Optional[str] = None
    updated_at: Optional[str] = None

class CRMAccountRecord(BaseModel):
    id: int
    customer_name: str
    tier: str  # Enterprise, Starter, Growth
    mrr: int
    account_manager: str
    status: str = "Active"
    created_at: str

class SupportTicketCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    customer: str = Field(..., min_length=1)
    priority: str = Field(default="Medium", pattern=r"^(High|Medium|Low)$")
    status: str = Field(default="Open", pattern=r"^(Open|Resolved|Closed)$")
    source_reference: Optional[str] = None
    summary: str = Field(..., min_length=1)

class SupportTicketRecord(BaseModel):
    id: int
    ticket_id: str
    customer: str
    priority: str
    status: str
    source_reference: Optional[str] = None
    summary: str
    created_at: str

class DocumentItem(BaseModel):
    id: int
    filename: str
    filepath: str
    title: str
    doc_type: str
    company: Optional[str] = None
    doc_date: Optional[str] = None
    content_preview: Optional[str] = None
    created_at: str
