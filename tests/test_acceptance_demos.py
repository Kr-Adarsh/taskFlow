"""
Deterministic Runtime Pipeline Tests for Demo 1 (Invoice) and Demo 2 (Support Ticket).
Validates executor state transitions, tool execution contracts, fault recovery mechanics,
and independent verifier enforcement using FakeProvider for rapid, deterministic CI coverage.
(Real-model autonomous acceptance testing is performed separately in tests/test_real_acceptance.py).
"""

from pathlib import Path
import pytest

from backend.app.agent.schemas import AgentDecision, AgentActionType, TaskPlan
from backend.app.agent.provider import FakeProvider
from backend.app.agent.loop import AgentRunner
from backend.app.agent.verifier import VerifierEngine
from backend.app.agent.schemas import VerificationIntent, SummaryAssessment
from backend.app.tools.registry import build_default_tool_registry
from backend.app.workspace.seed import seed_workspace, reset_demo_env
from backend.app.workspace.fault_injection import fault_manager
from backend.app.workspace.service import list_invoices, list_support_tickets

@pytest.fixture
def test_env(tmp_path: Path):
    db_file = tmp_path / "acceptance_workspace.db"
    seed_workspace(db_path=db_file, force_reseed=True)
    yield db_file

@pytest.mark.anyio
async def test_demo_1_invoice_end_to_end_with_fault_recovery(test_env: Path):
    """
    Demo 1 Acceptance Test:
    Objective: 'Find the latest invoice from Acme Corp, extract the amount and due date, enter it into Finance, and verify that it was recorded correctly.'
    Fault Injection: Armed for first write -> fails with 503 -> model observes transient error -> retries -> succeeds -> verified.
    """
    # Arm fault injection on database service
    fault_manager.reset()
    fault_manager.arm("finance_create_invoice", count=1)

    objective = "Find the latest invoice from Acme Corp, extract the amount and due date, enter it into Finance, and verify that it was recorded correctly."

    # Scripted decision flow representing autonomous agent steps
    fake_provider = FakeProvider([
        # 1. Planner
        TaskPlan(
            objective=objective,
            success_criteria=[
                "Latest Acme Corp invoice identified by date (INV-1044, 2026-09-15)",
                "Amount INR 84,500 and due date 2026-10-15 extracted",
                "Invoice entered into Finance system",
                "Verified against Finance database"
            ],
            strategy=["Search documents", "Read latest invoice", "Submit to Finance", "Verify"]
        ),
        # 2. Step 1: Search documents
        AgentDecision(
            thought="Searching document library for Acme Corp invoices.",
            action=AgentActionType.ACT,
            tool_name="document_search",
            tool_args={"query": "Acme Corp"}
        ),
        # 3. Step 2: Read latest invoice acme_invoice_1044.pdf (dated 2026-09-15)
        AgentDecision(
            thought="Identified latest invoice acme_invoice_1044.pdf by date (2026-09-15). Reading full content.",
            action=AgentActionType.ACT,
            tool_name="document_read",
            tool_args={"filename": "acme_invoice_1044.pdf"}
        ),
        # 4. Step 3: Open Finance page
        AgentDecision(
            thought="Extracted INV-1044, amount INR 84,500, due 2026-10-15. Navigating to Finance workspace.",
            action=AgentActionType.ACT,
            tool_name="browser_open",
            tool_args={"url": "/workspace/finance"}
        ),
        # 5. Step 4: First attempt to submit invoice (will trigger 503 fault injection)
        AgentDecision(
            thought="Submitting invoice INV-1044 details into Finance.",
            action=AgentActionType.ACT,
            tool_name="browser_type",
            tool_args={"element_id": "@company", "text": "Acme Corp"}
        ),
        # 6. Step 5: Type invoice number
        AgentDecision(
            thought="Typing invoice number.",
            action=AgentActionType.ACT,
            tool_name="browser_type",
            tool_args={"element_id": "@invoice_number", "text": "INV-1044"}
        ),
        # 7. Step 6: Type amount
        AgentDecision(
            thought="Typing amount.",
            action=AgentActionType.ACT,
            tool_name="browser_type",
            tool_args={"element_id": "@amount", "text": "84,500"}
        ),
        # 8. Step 7: Type due date
        AgentDecision(
            thought="Typing due date.",
            action=AgentActionType.ACT,
            tool_name="browser_type",
            tool_args={"element_id": "@due_date", "text": "2026-10-15"}
        ),
        # 9. Step 8: Click submit (receives 503 error feedback)
        AgentDecision(
            thought="Clicking record invoice submit button.",
            action=AgentActionType.ACT,
            tool_name="browser_click",
            tool_args={"element_id": "@submit_invoice"}
        ),
        # 10. Step 9: Model observes 503 transient lock error, decides to retry
        AgentDecision(
            thought="Observed 503 transient lock timeout in Finance gateway. Retrying submission.",
            action=AgentActionType.ACT,
            tool_name="browser_click",
            tool_args={"element_id": "@submit_invoice"}
        ),
        # 11. Step 10: Submission succeeds, agent requests independent verification
        AgentDecision(
            thought="Invoice recorded successfully. Requesting independent verification.",
            action=AgentActionType.READY_FOR_VERIFICATION
        )
    ])

    registry = build_default_tool_registry()
    verifier = VerifierEngine(db_path=test_env, intent=VerificationIntent(collection="invoices", company="Acme Corp", selection="latest"))

    # Let's mock browser tool responses with standard ToolResults for direct unit testing of loop transitions
    async def mock_browser_open(url: str):
        from backend.app.tools.base import ToolResult
        return ToolResult(ok=True, data={"url": url, "title": "Finance"})

    async def mock_browser_type(element_id: str, text: str, clear: bool = True):
        from backend.app.tools.base import ToolResult
        return ToolResult(ok=True, data={"element": element_id, "entered_text": text})

    call_count = {"clicks": 0}
    async def mock_browser_click(element_id: str):
        from backend.app.tools.base import ToolResult
        from backend.app.workspace.models import InvoiceCreate
        from backend.app.workspace.service import create_invoice
        call_count["clicks"] += 1
        if call_count["clicks"] == 1:
            # First attempt triggers fault injection
            try:
                fault_manager.maybe_fail("finance_create_invoice")
            except Exception as e:
                return ToolResult(ok=False, error=str(e), error_code="TRANSIENT_503_ERROR", retriable=True)
        
        # Second attempt succeeds and persists record
        inv = create_invoice(
            InvoiceCreate(
                company="Acme Corp",
                invoice_number="INV-1044",
                amount="84,500",
                due_date="2026-10-15",
                source_reference="acme_invoice_1044.pdf"
            ),
            db_path=test_env
        )
        return ToolResult(ok=True, data={"success_message": f"Invoice {inv.invoice_number} recorded", "id": inv.id})

    registry.register("browser_open", "open", registry.get("browser_open").parameters, mock_browser_open)
    registry.register("browser_type", "type", registry.get("browser_type").parameters, mock_browser_type)
    registry.register("browser_click", "click", registry.get("browser_click").parameters, mock_browser_click)

    runner = AgentRunner(
        provider=fake_provider,
        tool_registry=registry,
        verifier=verifier,
        max_steps=15,
        db_path=test_env
    )

    result = await runner.execute_task(objective)
    assert result["status"] == "completed"
    assert result["verification"]["verified"] is True
    assert "INV-1044" in str(result["verification"]["criteria_results"])

    # Verify database has exactly ONE persisted record for INV-1044
    invoices = [i for i in list_invoices(db_path=test_env) if i.invoice_number == "INV-1044"]
    assert len(invoices) == 1
    assert invoices[0].amount_minor == 8450000

@pytest.mark.anyio
async def test_demo_2_support_enterprise_positive_end_to_end(test_env: Path):
    """
    Demo 2 Acceptance Test (Enterprise Branch):
    Objective: 'Read complaint 4821, identify the customer, check their CRM account, and if they are an Enterprise customer, create a high-priority support ticket summarizing the complaint.'
    Customer: Acme Corp (Enterprise) -> creates high-priority ticket -> verified.
    """
    objective = "Read complaint 4821, identify the customer, check their CRM account, and if they are an Enterprise customer, create a high-priority support ticket summarizing the complaint."

    fake_provider = FakeProvider([
        # 1. Plan
        TaskPlan(
            objective=objective,
            success_criteria=[
                "Read complaint 4821",
                "Identify customer (Acme Corp)",
                "Verify CRM account tier is Enterprise",
                "Create high-priority support ticket summarizing complaint",
                "Verify ticket recorded"
            ],
            strategy=["Read complaint", "Check CRM", "Create Ticket", "Verify"]
        ),
        # 2. Step 1: Read complaint 4821
        AgentDecision(
            thought="Reading complaint 4821 to identify the customer and issue.",
            action=AgentActionType.ACT,
            tool_name="document_read",
            tool_args={"filename": "complaint_4821.txt"}
        ),
        # 3. Step 2: Open CRM
        AgentDecision(
            thought="Customer identified as Acme Corp. Checking CRM account tier.",
            action=AgentActionType.ACT,
            tool_name="browser_open",
            tool_args={"url": "/workspace/crm"}
        ),
        # 4. Step 3: Open Support page (tier verified as Enterprise)
        AgentDecision(
            thought="Acme Corp is verified as an Enterprise customer. Navigating to Support to create high-priority ticket.",
            action=AgentActionType.ACT,
            tool_name="browser_open",
            tool_args={"url": "/workspace/support"}
        ),
        # 5. Step 4: Submit Ticket
        AgentDecision(
            thought="Creating High priority ticket for Acme Corp summarizing EU sync outage.",
            action=AgentActionType.ACT,
            tool_name="browser_click",
            tool_args={"element_id": "@submit_ticket"}
        ),
        # 6. Step 5: Ready for verification
        AgentDecision(
            thought="Ticket created in Support system. Ready for verification.",
            action=AgentActionType.READY_FOR_VERIFICATION
        )
    ])

    registry = build_default_tool_registry()
    verifier = VerifierEngine(db_path=test_env, intent=VerificationIntent(collection="tickets", complaint_id="4821", priority="High", condition_tier="Enterprise"), provider=FakeProvider([SummaryAssessment(accurate=True, reason="Grounded", source_quotes=["severe database synchronization outages"], contradictions=[])]))

    async def mock_browser_open(url: str):
        from backend.app.tools.base import ToolResult
        return ToolResult(ok=True, data={"url": url, "title": "Workspace"})

    async def mock_browser_click(element_id: str):
        from backend.app.tools.base import ToolResult
        from backend.app.workspace.models import SupportTicketCreate
        from backend.app.workspace.service import create_support_ticket
        ticket = create_support_ticket(
            SupportTicketCreate(
                customer="Acme Corp",
                priority="High",
                source_reference="complaint_4821",
                summary="Severe database synchronization outage across EU servers"
            ),
            db_path=test_env
        )
        return ToolResult(ok=True, data={"ticket_id": ticket.ticket_id})

    registry.register("browser_open", "open", registry.get("browser_open").parameters, mock_browser_open)
    registry.register("browser_click", "click", registry.get("browser_click").parameters, mock_browser_click)

    runner = AgentRunner(
        provider=fake_provider,
        tool_registry=registry,
        verifier=verifier,
        max_steps=10,
        db_path=test_env
    )

    result = await runner.execute_task(objective)
    assert result["status"] == "completed"
    assert result["verification"]["verified"] is True
    assert "Acme Corp" in str(result["verification"]["criteria_results"])

    tickets = list_support_tickets(customer="Acme Corp", db_path=test_env)
    assert len(tickets) == 1
    assert tickets[0].priority == "High"

@pytest.mark.anyio
async def test_demo_2_support_non_enterprise_no_op_branch(test_env: Path):
    """
    Demo 2 Non-Enterprise Conditional Branch:
    Customer Beta Retail (complaint 4822) is Starter tier.
    Agent verifies tier is Starter -> Takes NO ticket mutation -> verifier confirms no-op success!
    """
    objective = "Read complaint 4822, identify the customer, check their CRM account, and if they are an Enterprise customer, create a high-priority support ticket summarizing the complaint."

    fake_provider = FakeProvider([
        TaskPlan(
            objective=objective,
            success_criteria=[
                "Read complaint 4822",
                "Identify customer (Beta Retail)",
                "Check CRM tier",
                "If not Enterprise, create no ticket and finish"
            ],
            strategy=["Read complaint", "Check CRM", "Conditional Branch", "Verify"]
        ),
        # Step 1: Read complaint 4822
        AgentDecision(
            thought="Reading complaint 4822.",
            action=AgentActionType.ACT,
            tool_name="document_read",
            tool_args={"filename": "complaint_4822.txt"}
        ),
        # Step 2: Open CRM
        AgentDecision(
            thought="Customer is Beta Retail. Checking CRM tier.",
            action=AgentActionType.ACT,
            tool_name="browser_open",
            tool_args={"url": "/workspace/crm"}
        ),
        # Step 3: Conclude that Beta Retail is Starter (non-Enterprise), so no ticket should be created
        AgentDecision(
            thought="Beta Retail is on the Starter tier (not Enterprise). As per instructions, no support ticket should be created. Requesting verification.",
            action=AgentActionType.READY_FOR_VERIFICATION
        )
    ])

    registry = build_default_tool_registry()
    verifier = VerifierEngine(db_path=test_env, intent=VerificationIntent(collection="tickets", complaint_id="4822", priority="High", condition_tier="Enterprise"))

    async def mock_browser_open(url: str):
        from backend.app.tools.base import ToolResult
        return ToolResult(ok=True, data={"url": url, "title": "CRM", "page_text_summary": "Beta Retail - Starter"})

    registry.register("browser_open", "open", registry.get("browser_open").parameters, mock_browser_open)

    runner = AgentRunner(
        provider=fake_provider,
        tool_registry=registry,
        verifier=verifier,
        max_steps=10,
        db_path=test_env
    )

    result = await runner.execute_task(objective)
    assert result["status"] == "completed"
    assert result["verification"]["verified"] is True
    assert "no_op" in str(result["verification"]["criteria_results"])

    # Confirm NO ticket was created in database
    tickets = list_support_tickets(customer="Beta Retail", db_path=test_env)
    assert len(tickets) == 0
