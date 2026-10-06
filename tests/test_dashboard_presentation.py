"""Dashboard presentation and existing endpoint wiring, without model requests."""
import os
import pytest
from playwright.async_api import async_playwright, expect
from test_real_acceptance import real_workspace
from backend.app.api.runs import set_runner_factory, broadcast_event
from backend.app.agent.provider import FakeProvider
from backend.app.agent_v2.graph import AgentRunner
from backend.app.agent.verifier import snapshot_state, state_delta
from backend.app.tools.browser_tools import browser_manager
from backend.app.workspace.fault_injection import fault_manager

pytestmark = pytest.mark.anyio


@pytest.fixture
async def dashboard_page(real_workspace):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            executable_path=os.getenv('TASKFLOW_CHROME_PATH', '/usr/bin/google-chrome'),
            headless=True, args=['--no-sandbox'],
        )
        page = await browser.new_page(viewport={'width': 1440, 'height': 1000})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        await page.goto(browser_manager.base_url + '/dashboard')
        try:
            yield page, errors
        finally:
            await browser.close()
        assert not errors


async def test_theme_persistence_and_existing_workspace_tabs(dashboard_page):
    page, _ = dashboard_page
    assert await page.locator('html').get_attribute('data-theme') == 'light'
    assert await page.locator('#objective_input').input_value() == ''
    assert await page.locator('#btn_execute').is_disabled()
    frame = page.frame_locator('#ws_iframe')
    await frame.locator('#company').wait_for()
    light = await frame.locator('body').evaluate('(body) => getComputedStyle(body).backgroundColor')
    await page.locator('#theme_toggle').click()
    assert await page.locator('html').get_attribute('data-theme') == 'dark'
    dark = await frame.locator('body').evaluate('(body) => getComputedStyle(body).backgroundColor')
    assert dark != light
    await page.reload()
    assert await page.locator('html').get_attribute('data-theme') == 'dark'
    for tab, control in [('crm', '#search_crm'), ('support', '#ticket_customer'), ('documents', '#btn_view_complaint_4821\\.txt')]:
        await page.locator('#tab_btn_' + tab).click()
        await frame.locator(control).wait_for()
        assert await frame.locator('html').get_attribute('data-theme') == 'dark'
        assert await frame.locator('body').evaluate('(body) => getComputedStyle(body).backgroundColor') == dark
    await page.locator('#tab_btn_finance').click()
    await frame.locator('#company').wait_for()


async def test_standalone_pages_and_real_agent_browser_use_the_shared_theme(dashboard_page):
    page, _ = dashboard_page
    await page.locator('#theme_toggle').click()
    await page.wait_for_function("document.documentElement.dataset.theme === 'dark'")
    response = await page.request.post(browser_manager.base_url + '/api/ui/theme', data={'theme': 'dark'})
    assert response.ok
    try:
        opened = await browser_manager.browser_open('/workspace/finance')
        assert opened.ok and opened.evidence['screenshot']
        agent = await browser_manager.get_page()
        assert await agent.locator('html').get_attribute('data-theme') == 'dark'
        controls_before = {item['id'] for item in opened.data['inspection']['interactive_elements']}
        assert '@company' in controls_before and '@submit_invoice' in controls_before
        await agent.locator('#company').fill('Example Co')
        await browser_manager.set_ui_theme('light')
        assert await agent.locator('html').get_attribute('data-theme') == 'light'
        assert await agent.locator('#company').input_value() == 'Example Co'
        inspected = await browser_manager.browser_inspect()
        assert controls_before == {item['id'] for item in inspected.data['interactive_elements']}
        # New navigation uses the same CSS as both the preview and genuine captures.
        opened = await browser_manager.browser_open('/workspace/support')
        assert opened.ok
        assert await agent.locator('html').get_attribute('data-theme') == 'light'
        assert await agent.locator('link[rel="stylesheet"]').get_attribute('href') == '/static/theme.css'
        assert await agent.locator('style').count() == 0
        invalid = await page.request.post(browser_manager.base_url + '/api/ui/theme', data={'theme': 'other'})
        assert invalid.status == 422
    finally:
        await browser_manager.close()
        await browser_manager.set_ui_theme('light')


async def test_step_events_wrap_and_do_not_render_task_html(dashboard_page):
    page, _ = dashboard_page
    unsafe = '<img src=x onerror=alert(1)>'
    thought = 'Observed evidence. ' * 40 + unsafe
    events = [
        {'id': 1, 'event_type': 'MODEL_CALL', 'payload': {'stage': 'planner', 'metadata': {'provider': 'gemini', 'model': 'gemini-3.5-flash-lite'}}},
        {'id': 2, 'event_type': 'DECISION', 'payload': {'step': 1, 'task_id': 't', 'decision': {'thought': thought, 'action': 'act', 'tool_name': 'browser_type', 'tool_args': {'element_id': '@entry', 'text': 'x' * 2000}}}},
        {'id': 3, 'event_type': 'ACTION', 'payload': {'step': 1, 'task_id': 't', 'tool': 'browser_type', 'args': {'element_id': '@entry', 'text': 'x' * 2000}}},
        {'id': 4, 'event_type': 'OBSERVATION', 'payload': {'step': 1, 'task_id': 't', 'ok': False, 'error': 'Transient service failure', 'retriable': True, 'evidence': {'http_status': 503, 'commit_state': 'not_committed'}}},
    ]
    await page.evaluate('(events) => events.forEach(handleEvent)', events)
    assert 'Gemini' in await page.locator('#model_label').inner_text()
    assert await page.locator('.timeline-step').count() == 1
    assert await page.locator('#decision_count').inner_text() == '1'
    assert await page.locator('#action_count').inner_text() == '1'
    assert await page.locator('#event_count').inner_text() == '4 events'
    assert await page.locator('#timeline_stream img').count() == 0
    assert 'Reported as safe to retry' in await page.locator('#timeline_stream').inner_text()
    await page.evaluate('handleEvent', events[-1])
    assert await page.locator('#event_count').inner_text() == '4 events'
    await page.set_viewport_size({'width': 390, 'height': 844})
    await page.locator('.timeline-step summary').click()
    assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    await page.locator('#timeline_mode').click()
    assert await page.locator('#timeline_stream > .timeline-item').count() == 4
    assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth')


async def test_failure_reports_show_actual_provider_and_verified_status(dashboard_page):
    page, _ = dashboard_page
    await page.evaluate('renderReport', {
        'status': 'failed', 'provider': 'GeminiProvider', 'model': 'gemini-3.5-flash-lite',
        'duration_seconds': 60.2, 'steps': 0, 'error': 'Gemini transport failed or request timed out',
        'usage': {'requests': 1, 'prompt_tokens': 0, 'completion_tokens': 0, 'incomplete': True},
    })
    assert await page.locator('#system_status').inner_text() == 'Failed'
    assert 'Gemini' in await page.locator('#model_label').inner_text()
    assert 'gpt-oss' not in await page.locator('#model_label').inner_text()
    assert 'usage incomplete' in await page.locator('#final_report').inner_text()
    assert await page.locator('#verif_status').text_content() == 'Not verified'
    await page.evaluate('renderReport', {
        'status': 'waiting_for_clarification', 'duration_seconds': 10, 'steps': 4,
        'question': 'Verification interpretation is missing an exact source identifier',
        'tasks': [{'result': {'metrics': {'region': 'South', 'decline': 27000}}}],
    })
    assert await page.locator('#system_status').inner_text() == 'Needs clarification'
    assert 'not be resolved' in await page.locator('#final_report').inner_text()
    assert 'South' in await page.locator('#final_report').inner_text()
    assert await page.locator('#final_report').get_by_role('button', name='Edit objective').count() == 1
    await page.locator('#final_report').get_by_role('button', name='Edit objective').click()
    assert await page.locator('#objective_input').evaluate('(input) => input === document.activeElement')


async def test_interpretation_block_is_distinct_from_a_user_clarification(dashboard_page):
    page, _ = dashboard_page
    verification = {'verified': False, 'summary': 'Verification contract remained invalid',
                    'evidence': {'interpretation_failure': {'origin': 'verifier_interpretation',
                                                           'code': 'VERIFICATION_CONTRACT_INVALID'}}}
    await page.evaluate('renderReport', {'status': 'failed', 'verification': verification, 'steps': 8})
    assert await page.locator('#system_status').inner_text() == 'Verification blocked'
    assert await page.locator('#verif_status').text_content() == 'Blocked'
    assert 'more information' in await page.locator('#final_report').inner_text()
    assert await page.locator('#final_report').get_by_role('button', name='Edit objective').count() == 0
    # Terminal events must preserve the typed classification, including on replay.
    await page.evaluate('handleEvent', {'event_type': 'STREAM_END', 'payload': {'status': 'failed'}})
    assert await page.locator('#system_status').inner_text() == 'Verification blocked'
    await page.evaluate('clearPresentation()')
    await page.evaluate('renderReport', {'status': 'waiting_for_clarification', 'question': 'Which of the two sources should be used?'})
    assert await page.locator('#system_status').inner_text() == 'Needs clarification'


async def test_recovered_python_error_is_retained_without_being_current_failure(dashboard_page):
    page, _ = dashboard_page
    error = {'step': 2, 'code': 'PYTHON_RESULT_ERROR', 'error': 'Named tables must be pandas DataFrames'}
    task = {'task_id': 'data', 'goal': 'Compare totals', 'status': 'FAILED', 'errors': [error]}
    await page.evaluate('renderTasks', [task])
    assert await page.locator('#task_graph .task-error').count() == 1
    await page.evaluate('handleEvent', {'event_type': 'OBSERVATION', 'payload': {
        'task_id': 'data', 'step': 3, 'ok': True, 'data': {'stage': 'full'}}})
    await page.evaluate('renderTasks', [task])
    assert await page.locator('#task_graph .task-error').count() == 0
    assert 'Recovered attempt' in await page.locator('#task_graph').inner_text()
    await page.locator('#task_graph details > summary').click()
    assert error['error'] in await page.locator('#task_graph').inner_text()


async def test_independent_calculation_is_readable_and_not_inferred_as_pass(dashboard_page):
    page, _ = dashboard_page
    verification={'verified':True,'evidence':{
        'contract':{'document_id':'sales.csv','group_column':'region','group_metric':'region','value_metric':'decline',
                    'measure':'difference','aggregate':'sum','value_column':'revenue','baseline_period':'2026-08','current_period':'2026-09','convention':'baseline_minus_current'},
        'expected':{'group':'South','value':27000,'full_dataset_rows':800,'groups_compared':4,
                    'group_values_preview':{'East':4000,'North':5000,'South':27000,'West':-5000}},
        'actual':{'region':'South','decline':27000}}}
    await page.evaluate('renderVerification',verification)
    proof=page.locator('.calculation-proof')
    assert '800 rows checked' in await proof.inner_text()
    assert '2026-08 minus 2026-09' in await proof.inner_text()
    assert await proof.locator('tbody tr').count()==4
    assert 'South' in await proof.locator('.selected-result').inner_text()
    assert '27,000' in await proof.inner_text()
    await page.set_viewport_size({'width':390,'height':844})
    assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    verification['verified']=False
    verification['evidence']['actual']={'region':'East','decline':1}
    await page.evaluate('renderVerification',verification)
    assert 'has not passed' in await proof.inner_text()
    assert await page.locator('#verif_status').text_content()=='Not verified'


async def test_invoice_run_streams_recovery_verification_and_saved_records(dashboard_page):
    page, _ = dashboard_page
    await page.evaluate('setPreset(1)')
    objective = await page.locator('#objective_input').input_value()
    responses = [
        {'objective': objective, 'success_criteria': [objective], 'tasks': [
            {'task_id': 'invoice', 'goal': objective, 'success_criteria': [objective],
             'verification_capability': 'browser'}]},
        {'action': 'act', 'thought': 'Read the source', 'tool_name': 'read_document_chunks',
         'tool_args': {'document_id': 'acme_invoice_1044.pdf'}},
        {'action': 'act', 'thought': 'Open Finance', 'tool_name': 'browser_open',
         'tool_args': {'url': '/workspace/finance'}},
    ]
    for field, value in [('company', 'Acme Corp'), ('invoice_number', 'INV-1044'),
                         ('amount', '84500'), ('due_date', '2026-10-15'),
                         ('source_reference', 'acme_invoice_1044.pdf')]:
        responses.append({'action': 'act', 'thought': 'Fill the observed field',
                          'tool_name': 'browser_type',
                          'tool_args': {'element_id': '@' + field, 'text': value}})
    for thought in ('Submit the invoice', 'Retry the uncommitted submit'):
        responses.append({'action': 'act', 'thought': thought, 'tool_name': 'browser_click',
                          'tool_args': {'element_id': '@submit_invoice'}})
    responses.append({'collection': 'invoices', 'company': 'Acme Corp', 'selection': 'latest'})
    provider = FakeProvider(responses)
    before = snapshot_state()
    fault_manager.arm('finance_create_invoice', 1)
    set_runner_factory(lambda max_steps=20: AgentRunner(
        provider=provider, max_steps=max_steps, event_callback=broadcast_event))
    try:
        await page.locator('#btn_execute').click()
        await expect(page.locator('#system_status')).to_have_text('Completed', timeout=30000)
        await expect(page.locator('#verif_status')).to_have_text('Passed')
        assert await page.locator('.timeline-step').count() == 9
        assert await page.locator('#action_count').inner_text() == '9'
        assert 'safe to retry' in await page.locator('#timeline_stream').inner_text()
        frame = page.frame_locator('#ws_iframe')
        await expect(frame.locator('table')).to_contain_text('INV-1044')
        assert await frame.locator('form').get_attribute('action') == '/workspace/finance/submit'
        assert not provider.responses
        delta = state_delta(before, snapshot_state())
        assert len(delta['invoices']['created']) == 1
        assert delta['invoices']['created'][0]['amount_minor'] == 8450000
        assert not any(changes for collection, values in delta.items()
                       for kind, changes in values.items()
                       if collection != 'invoices' or kind != 'created')
        await page.locator('#toggle_view_btn').click()
        await expect(page.locator('#agent_screenshot')).to_be_visible()
        await page.reload()
        await expect(page.locator('#system_status')).to_have_text('Completed')
        assert await page.locator('.timeline-step').count() == 9
    finally:
        set_runner_factory(None)


async def test_evidence_and_intercepted_actions_are_presented_truthfully(dashboard_page):
    page, _ = dashboard_page
    await page.evaluate('renderMemory', {
        'dataset_profiles': {'sample.csv': {'shape': [800, 3], 'sample_rows': []}},
        'pages': {'/workspace/accounts': {'tables': [
            [['Customer', 'Tier'], ['Example Co', 'Gold']]]}},
    })
    await page.locator('.disclosure > summary').click()
    assert '800 rows' in await page.locator('#memory_grid').inner_text()
    assert '1 observed tables' in await page.locator('#memory_grid').inner_text()
    await page.evaluate('(events) => events.forEach(handleEvent)', [
        {'event_type': 'DECISION', 'payload': {'step': 1, 'task_id': 't',
         'decision': {'thought': 'Revisit a page', 'action': 'act',
                      'tool_name': 'browser_open', 'tool_args': {'url': '/workspace/crm'}}}},
        {'event_type': 'OBSERVATION', 'payload': {'step': 1, 'task_id': 't', 'ok': False,
         'error': 'OSCILLATION_DETECTED: repeated state', 'error_code': 'OSCILLATION_DETECTED'}},
    ])
    assert 'Not executed' in await page.locator('.timeline-step').inner_text()
    assert await page.locator('#action_count').inner_text() == '0'


async def test_demo_tools_use_existing_routes_and_remain_optional(dashboard_page):
    page, _ = dashboard_page
    requests = []
    page.on('request', lambda request: requests.append((request.method, request.url, request.post_data)))
    await page.locator('.tools > summary').click()
    await page.locator('#btn_arm_fault').click()
    await expect(page.locator('#btn_arm_fault')).to_have_text('Recovery test armed')
    assert any(method == 'POST' and '/api/workspace/fault-injection' in url and 'finance_create_invoice' in body for method, url, body in requests)
    await page.locator('#btn_reset').click()
    await expect(page.locator('#btn_arm_fault')).to_have_text('Test one failed submit')
    assert any(method == 'POST' and url.endswith('/api/workspace/reset') for method, url, _ in requests)
    assert not any(method == 'POST' and url.endswith('/api/runs') for method, url, _ in requests)


async def test_visual_layout_at_desktop_and_mobile(dashboard_page, tmp_path):
    page, _ = dashboard_page
    output = tmp_path
    await page.frame_locator('#ws_iframe').locator('#company').wait_for()
    await page.screenshot(path=str(output / 'dashboard-light.png'), full_page=True, animations='disabled')
    await page.locator('#theme_toggle').click()
    await page.mouse.move(0, 0)
    await page.screenshot(path=str(output / 'dashboard-dark.png'), full_page=True, animations='disabled')
    await page.set_viewport_size({'width': 390, 'height': 844})
    await page.locator('#theme_toggle').click()
    assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    frame = page.frame_locator('#ws_iframe')
    assert await frame.locator('#company').evaluate('(input) => input.getBoundingClientRect().width') > 250
    await page.mouse.move(0, 0)
    await page.screenshot(path=str(output / 'dashboard-mobile.png'), full_page=True, animations='disabled')
