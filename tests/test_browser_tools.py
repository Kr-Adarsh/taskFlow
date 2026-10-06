"""
Integration tests for Playwright browser tools driving actual workspace web pages.
Exercises page opening, semantic inspection, typing, clicking, and server error detection.
"""

import threading
import time
import pytest
import uvicorn

from backend.app.main import app
from backend.app.workspace.seed import reset_demo_env
from backend.app.workspace.fault_injection import fault_manager
from backend.app.tools.browser_tools import browser_manager
from backend.app.tools.registry import build_default_tool_registry

TEST_SERVER_PORT = 8765
TEST_SERVER_URL = f"http://127.0.0.1:{TEST_SERVER_PORT}"

class UvicornTestServer(threading.Thread):
    def __init__(self, app, port):
        super().__init__(daemon=True)
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        self.server = uvicorn.Server(config)

    def run(self):
        self.server.run()

    def stop(self):
        self.server.should_exit = True

@pytest.fixture(scope="module", autouse=True)
def run_test_server():
    reset_demo_env()
    server = UvicornTestServer(app, TEST_SERVER_PORT)
    server.start()
    time.sleep(1.0)  # Wait for uvicorn to bind
    browser_manager.base_url = TEST_SERVER_URL
    try:
        yield
    finally:
        server.stop()
        server.join(timeout=5)

@pytest.fixture(autouse=True)
async def reset_state():
    reset_demo_env()
    try:
        yield
    finally:
        await browser_manager.close()

@pytest.mark.anyio
async def test_browser_finance_workflow_end_to_end():
    registry = build_default_tool_registry()

    # 1. Open Finance page
    res_open = await registry.execute("browser_open", {"url": "/workspace/finance"})
    assert res_open.ok is True
    assert "Finance" in res_open.data["title"]

    # 2. Inspect page
    res_inspect = await registry.execute("browser_inspect", {})
    assert res_inspect.ok is True
    elements = res_inspect.data["interactive_elements"]
    element_ids = [el["id"] for el in elements]
    assert "@company" in element_ids
    assert "@invoice_number" in element_ids
    assert "@amount" in element_ids
    assert "@due_date" in element_ids
    assert "@submit_invoice" in element_ids

    # 3. Type into fields
    await registry.execute("browser_type", {"element_id": "@company", "text": "Acme Corp"})
    await registry.execute("browser_type", {"element_id": "@invoice_number", "text": "INV-1044"})
    await registry.execute("browser_type", {"element_id": "@amount", "text": "84,500"})
    await registry.execute("browser_type", {"element_id": "@due_date", "text": "2026-10-15"})

    await registry.execute("browser_type", {"element_id": "@source_reference", "text": "acme_invoice_1044.pdf"})
    # 4. Click Submit
    res_click = await registry.execute("browser_click", {"element_id": "@submit_invoice"})
    assert res_click.ok is True
    assert res_click.data["outcome"] == "confirmed_mutation"
    assert "recorded successfully" in res_click.data.get("success_message", "")

    # 5. Verify row is visible in table
    inspect_after = await registry.execute("browser_inspect", {})
    assert "INV-1044" in inspect_after.data["page_text_summary"]

    await browser_manager.close()

@pytest.mark.anyio
async def test_browser_surfaces_503_error_and_recovers():
    registry = build_default_tool_registry()

    # Arm fault injection
    fault_manager.reset()
    fault_manager.arm("finance_create_invoice", count=1)

    # Open finance
    await registry.execute("browser_open", {"url": "/workspace/finance"})

    # Fill form
    await registry.execute("browser_type", {"element_id": "@company", "text": "Acme Corp"})
    await registry.execute("browser_type", {"element_id": "@invoice_number", "text": "INV-7777"})
    await registry.execute("browser_type", {"element_id": "@amount", "text": "55,000"})
    await registry.execute("browser_type", {"element_id": "@due_date", "text": "2026-10-20"})

    await registry.execute("browser_type", {"element_id": "@source_reference", "text": "acme_invoice_1044.pdf"})
    # Submit 1 -> Must fail with 503 and retriable=True
    click_res1 = await registry.execute("browser_click", {"element_id": "@submit_invoice"})
    assert click_res1.ok is False
    assert click_res1.retriable is True
    assert "Finance Service Unavailable" in click_res1.error
    assert click_res1.error_code == "TRANSIENT_503_ERROR"
    assert click_res1.evidence["http_status"] == 503

    # Submit 2 (retry) -> Must succeed
    click_res2 = await registry.execute("browser_click", {"element_id": "@submit_invoice"})
    assert click_res2.ok is True
    assert "recorded successfully" in click_res2.data.get("success_message", "")

    await browser_manager.close()

@pytest.mark.anyio
async def test_browser_crm_lookup():
    registry = build_default_tool_registry()

    # Open CRM
    await registry.execute("browser_open", {"url": "/workspace/crm"})
    inspect = await registry.execute("browser_inspect", {})
    assert "CRM" in inspect.data["title"]
    assert "Acme Corp" in inspect.data["page_text_summary"]
    assert "Enterprise" in inspect.data["page_text_summary"]

    # Search for Beta Retail
    await registry.execute("browser_type", {"element_id": "@search_crm", "text": "Beta Retail"})
    clicked = await registry.execute("browser_click", {"element_id": "@button_search_crm"})
    assert clicked.ok and clicked.data["outcome"] == "confirmed_navigation"

    inspect_search = await registry.execute("browser_inspect", {})
    assert "Beta Retail" in inspect_search.data["page_text_summary"]
    assert "Starter" in inspect_search.data["page_text_summary"]

    await browser_manager.close()

@pytest.mark.anyio
async def test_browser_rejects_external_urls():
    """
    Adversarial test 4:
    Browser tool must reject navigation to external HTTP/HTTPS URLs and non-workspace schemes.
    """
    registry = build_default_tool_registry()

    # 1. Attempt navigation to arbitrary public internet URL
    res_google = await registry.execute("browser_open", {"url": "https://google.com"})
    assert res_google.ok is False
    assert res_google.error_code == "RESTRICTED_URL"
    assert "forbidden" in res_google.error.lower()

    # 2. Attempt navigation to arbitrary HTTP domain
    res_evil = await registry.execute("browser_open", {"url": "http://evil-tracker.example/phish"})
    assert res_evil.ok is False
    assert res_evil.error_code == "RESTRICTED_URL"
    assert "forbidden" in res_evil.error.lower()

    # 3. Disallowed scheme
    res_js = await registry.execute("browser_open", {"url": "javascript:alert(1)"})
    assert res_js.ok is False
    assert res_js.error_code == "RESTRICTED_URL"

    # 4. Valid local workspace path must succeed
    res_local = await registry.execute("browser_open", {"url": "/workspace/finance"})
    assert res_local.ok is True
    assert "Finance" in res_local.data["title"]

    await browser_manager.close()


@pytest.mark.anyio
async def test_browser_hostile_origin_redirect_subresource_and_native_validation():
    registry=build_default_tool_registry()
    for url in [TEST_SERVER_URL+'@example.com/',TEST_SERVER_URL+'.evil/',TEST_SERVER_URL.replace('127.0.0.1','user:password@127.0.0.1'), '//example.com/', 'data:text/html,attack']:
        res=await registry.execute('browser_open',{'url':url})
        assert not res.ok and res.error_code=='RESTRICTED_URL'
    await registry.execute('browser_open',{'url':'/workspace/finance'})
    rejected=await registry.execute('browser_click',{'element_id':'@submit_invoice'})
    assert not rejected.ok and rejected.error_code=='FORM_PRECONDITION_FAILED'
    page=await browser_manager.get_page()
    # Even navigation initiated by page JavaScript is checked by the request boundary.
    await page.evaluate("location.href='http://127.0.0.1:8999/forbidden'")
    await page.wait_for_timeout(150)
    assert browser_manager._blocked
    assert all('forbidden' in url for url in browser_manager._blocked)

@pytest.mark.anyio
async def test_task_and_document_html_cannot_execute(tmp_path):
    from backend.app.workspace.db import get_db_connection
    attack='<img src=x onerror="window.unsafeExecuted=1"><script>window.unsafeExecuted=1</script>'
    document=tmp_path/'unsafe.txt'; document.write_text(attack)
    with get_db_connection() as conn:
        conn.execute("INSERT INTO documents_index(filename,filepath,title,doc_type,created_at) VALUES(?,?,?,?,?)",('unsafe.txt',str(document),attack,'text','now'))
        conn.commit()
    registry=build_default_tool_registry()
    result=await registry.execute('browser_open',{'url':'/workspace/documents?view=unsafe.txt'})
    assert result.ok
    page=await browser_manager.get_page()
    assert await page.evaluate('window.unsafeExecuted') is None
    assert await page.locator('#doc_full_text').inner_text()==attack
    await registry.execute('browser_open',{'url':'/dashboard'})
    await page.evaluate("""attack=>{handleEvent({event_type:'TASK',timestamp:new Date().toISOString(),payload:{objective:attack}}); renderMemory({sources:{source:attack}}); renderVerification({verified:false,summary:attack,criteria_results:[{criterion:attack,passed:false,evidence:{expected:attack}}]});}""",attack)
    assert await page.evaluate('window.unsafeExecuted') is None
    assert '<script>window.unsafeExecuted=1</script>' in await page.locator('#timeline_stream').inner_text()

from fastapi.responses import RedirectResponse, HTMLResponse

@app.get('/boundary-test-redirect')
def boundary_redirect():
    return RedirectResponse('http://127.0.0.1:8999/forbidden')

@app.get('/boundary-test-resource')
def boundary_resource():
    return HTMLResponse('<img src="http://127.0.0.1:8999/forbidden">')

@pytest.mark.anyio
async def test_redirect_and_subresource_cannot_leave_workspace():
    registry=build_default_tool_registry()
    for path in ['/boundary-test-redirect','/boundary-test-resource']:
        result=await registry.execute('browser_open',{'url':path})
        assert not result.ok and result.error_code=='RESTRICTED_URL', (path,result,browser_manager._blocked)

@pytest.mark.anyio
async def test_dispatch_without_observed_outcome_is_uncertain():
    registry=build_default_tool_registry()
    await registry.execute('browser_open',{'url':'/dashboard'})
    await registry.execute('browser_inspect',{})
    result=await registry.execute('browser_click',{'element_id':'@toggle_view_btn'})
    assert not result.ok and result.error_code=='UNCERTAIN_OUTCOME'
