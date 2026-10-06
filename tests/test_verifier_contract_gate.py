"""Opt-in checks for contract interpretation failures reproduced during acceptance."""
import os
import uuid

import pytest

from backend.app.agent.provider import get_default_provider
from backend.app.agent.verifier import VerifierEngine

pytestmark = [pytest.mark.anyio, pytest.mark.skipif(
    os.getenv('OPERON_RUN_REAL') != '1', reason='Explicit real Groq gate')]


async def test_real_crm_condition_is_supported_not_erased():
    provider = get_default_provider()
    verifier = VerifierEngine(provider=provider)
    intent = await verifier.interpret(
        'Read complaint 4821, identify the customer, check their CRM account, and if they are an Enterprise customer, create a high-priority support ticket summarizing the complaint.',
        ['Customer type confirmed as Enterprise', 'Ticket ID and creation timestamp recorded'],
    )
    assert intent.collection == 'tickets'
    assert intent.complaint_id == '4821' and intent.condition_tier == 'Enterprise'
    assert intent.priority == 'High' and intent.require_new_record
    assert not intent.unsupported_criteria


async def test_real_unseen_invoice_identity_and_selection_preserved():
    company = 'Cedar Harbor ' + uuid.uuid4().hex[:6]
    provider = get_default_provider()
    intent = await VerifierEngine(provider=provider).interpret(
        f'Find the latest invoice from {company}, extract the amount and due date, enter it into Finance, and verify that it was recorded correctly.',
        ['Latest requested invoice recorded accurately'],
    )
    assert intent.collection == 'invoices' and intent.company == company
    assert intent.selection == 'latest' and intent.require_new_record
    assert not intent.unsupported_criteria
