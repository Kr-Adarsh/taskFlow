"""Progress continuity through the real browser and scripted V2 decisions."""
import json

import pytest

from test_real_acceptance import real_workspace, INVOICE
from test_v2_integration import act, plan
from backend.app.agent.provider import FakeProvider
from backend.app.agent.verifier import snapshot_state, state_delta
from backend.app.agent_v2.context import ContextMemory, decision_prompt
from backend.app.agent_v2.graph import AgentRunner
from backend.app.agent_v2.state import Decision, GraphState, Subtask
from backend.app.capabilities.documents import read_document_chunks
from backend.app.capabilities.python.profile import dataset_profile
from backend.app.tools.base import ToolResult
from backend.app.tools.browser_tools import browser_manager
from backend.app.workspace.db import get_db_connection
from backend.app.workspace.fault_injection import fault_manager

pytestmark = pytest.mark.anyio


@pytest.fixture
async def browser_workspace(real_workspace, anyio_backend):
    yield real_workspace
    await browser_manager.close()


def context_data(messages):
    return json.loads(messages[-1]['content'].split('\n', 1)[1])['untrusted_task_data']


async def test_form_memory_survives_document_and_profile_updates(browser_workspace):
    memory = ContextMemory()
    opened = await browser_manager.browser_open('/workspace/finance')
    memory.update('browser_open', {'url': '/workspace/finance'}, opened)
    typed = await browser_manager.browser_type('@company', 'Acme Corp')
    assert typed.ok
    memory.update('browser_type', {'element_id': '@company', 'text': 'Acme Corp'}, typed)
    read = read_document_chunks('acme_invoice_1044.pdf')
    memory.update('read_document_chunks', {'document_id': 'acme_invoice_1044.pdf'}, read)
    memory.update('profile_dataset', {'document_id': 'sales.csv'},
                  ToolResult(ok=True, data=dataset_profile('sales.csv')))
    task = Subtask(task_id='form', goal='Complete observed form', success_criteria=['Recorded'], verification_capability='browser')
    state = GraphState(run_id='memory', objective=task.goal, tasks=[task], observation=read.model_dump())
    data = context_data(decision_prompt(state, task, memory, []))
    url = browser_manager.base_url + '/workspace/finance'
    assert data['current_browser_state']['url'] == url
    fields = {field['id']: field for field in data['current_browser_state']['interactive_elements']}
    assert fields['@company']['value'] == 'Acme Corp'
    assert fields['@source_reference']['required'] and fields['@source_reference']['value'] == ''
    assert 'INV-1044' in json.dumps(data)


async def test_same_page_open_keeps_values_and_query_navigation_is_distinct(browser_workspace, monkeypatch):
    await browser_manager.browser_open('/workspace/finance')
    await browser_manager.browser_type('@company', 'Example Customer')
    page = await browser_manager.get_page()
    original = page.goto
    navigations = []

    async def tracked(url, **kwargs):
        navigations.append(url)
        return await original(url, **kwargs)

    monkeypatch.setattr(page, 'goto', tracked)
    for url in ('/workspace/finance', browser_manager.base_url + '/workspace/finance#form'):
        result = await browser_manager.browser_open(url)
        assert result.ok and result.data['already_at_target'] and result.data['no_op']
        assert result.error_code == 'REDUNDANT_ACTION'
        assert await page.locator('#company').input_value() == 'Example Customer'
    assert navigations == []
    result = await browser_manager.browser_open('/workspace/finance?view=other')
    assert result.ok and len(navigations) == 1
    assert await page.locator('#company').input_value() == ''


async def test_redundant_type_does_not_dispatch_input_and_append_still_works(browser_workspace):
    await browser_manager.browser_open('/workspace/finance')
    await browser_manager.browser_type('@company', 'Acme Corp')
    page = await browser_manager.get_page()
    await page.evaluate("window.inputEvents=0; document.querySelector('#company').addEventListener('input',()=>window.inputEvents++)")
    result = await browser_manager.browser_type('@company', 'Acme Corp')
    assert result.ok and result.data['already_satisfied'] and result.data['no_op']
    assert result.error_code == 'REDUNDANT_ACTION'
    assert await page.evaluate('window.inputEvents') == 0
    result = await browser_manager.browser_type('@company', ' Europe', clear=False)
    assert result.ok and not result.data['no_op']
    assert await page.locator('#company').input_value() == 'Acme Corp Europe'
    assert await page.evaluate('window.inputEvents') == 1


async def test_generic_submit_preflight_checks_only_the_associated_enabled_form(browser_workspace):
    before = snapshot_state()
    await browser_manager.browser_open('/workspace/finance')
    page = await browser_manager.get_page()
    await page.set_content('''<form id="other"><input id="unrelated" required></form>
        <form id="contact" onsubmit="window.submits++; event.preventDefault()">
          <label for="recipient">Recipient email</label><input id="recipient" type="email" required>
          <input id="disabled_field" required disabled>
          <fieldset disabled><input id="disabled_group" required></fieldset>
          <button id="send" type="submit">Send</button>
        </form><script>window.submits=0</script>''')
    await browser_manager.browser_inspect()
    for value in ('', 'invalid-email'):
        if value:
            await browser_manager.browser_type('@recipient', value)
        result = await browser_manager.browser_click('@send')
        assert not result.ok and result.error_code == 'FORM_PRECONDITION_FAILED'
        assert result.evidence['commit_state'] == 'not_committed'
        assert [(field['id'], field['label']) for field in result.evidence['missing_required_fields']] == [('@recipient', 'Recipient email')]
        assert await page.evaluate('window.submits') == 0
        assert snapshot_state() == before
    await browser_manager.browser_type('@recipient', 'person@example.com')
    result = await browser_manager.browser_click('@send')
    assert result.error_code != 'FORM_PRECONDITION_FAILED'
    assert await page.evaluate('window.submits') == 1
    assert snapshot_state() == before


async def test_exact_reads_are_cached_but_changed_sources_and_other_chunks_are_read(browser_workspace, monkeypatch):
    source = browser_workspace / 'notes.txt'
    source.write_text('A' * 2200)
    with get_db_connection() as connection:
        connection.execute('INSERT INTO documents_index(filename,filepath,title,doc_type,created_at) VALUES(?,?,?,?,?)',
                           ('notes.txt', str(source), 'Notes', 'text', 'now'))
        connection.commit()
    runner = AgentRunner(provider=FakeProvider([]))
    runner.memory_mgr = ContextMemory()
    runner.source_versions = {}
    monkeypatch.setattr(runner, '_emit_event', lambda *args: None)
    original = runner.registry.execute
    calls = []

    async def tracked(name, args):
        calls.append((name, args))
        return await original(name, args)

    monkeypatch.setattr(runner.registry, 'execute', tracked)
    task = Subtask(task_id='notes', goal='Read notes', success_criteria=['Sourced'], verification_capability='documents')

    async def read(ids, **extras):
        args = {'document_id': 'notes.txt', 'chunk_ids': ids, **extras}
        state = GraphState(run_id='cache', objective=task.goal, tasks=[task], current_task_id=task.task_id,
                           decision=Decision.model_validate(act('read_document_chunks', **args)))
        update = await runner.execute(state)
        result = ToolResult.model_validate(update['tool_result'])
        runner.memory_mgr.update('read_document_chunks', args, result)
        return result

    first = await read(['notes.txt:1:1000'])
    assert first.ok and len(calls) == 1
    repeated = await read(['notes.txt:1:1000'])
    assert repeated.ok and repeated.data['already_available'] and repeated.data['no_op']
    assert repeated.error_code == 'REDUNDANT_ACTION' and len(calls) == 1
    assert repeated.data['chunks'] == first.data['chunks']
    assert runner.memory_mgr.no_progress_streak == 1
    assert (await read(['notes.txt:1:0'])).ok and len(calls) == 2
    ordered = await read(['notes.txt:1:1000', 'notes.txt:1:0'])
    assert [chunk['chunk_id'] for chunk in ordered.data['chunks']] == ['notes.txt:1:0', 'notes.txt:1:1000']
    invalid = await read(['notes.txt:1:0'], unexpected=True)
    assert not invalid.ok and invalid.error_code == 'INVALID_ARGUMENTS'
    source.write_text('B' * 2200)
    updated = await read(['notes.txt:1:1000'])
    assert updated.ok and updated.data['chunks'][0]['text'].startswith('B')
    assert not updated.data.get('already_available')


def test_compact_page_memory_is_bounded_and_retains_selected_values():
    memory = ContextMemory()
    for index in range(5):
        inspection = {'url': f'/form?record={index}', 'tables': [], 'page_text_summary': 'giant text' * 5000,
                      'interactive_elements': [{'id': f'@field{i}', 'tag': 'select', 'type': 'select-one', 'label': 'Label',
                                                'value': 'chosen', 'required': True, 'disabled': False,
                                                'options': [{'value': 'chosen', 'label': 'Chosen option'}]} for i in range(80)]}
        memory.update('browser_inspect', {}, ToolResult(ok=True, data=inspection))
    assert len(memory.pages) == 3 and len(memory.relevant()['historical_evidence']) == 2
    assert memory.current_url == '/form?record=4'
    page = memory.pages[memory.current_url]
    assert len(page['interactive_elements']) == 40
    assert page['interactive_elements'][0]['selected_option'] == {'value': 'chosen', 'label': 'Chosen option'}
    assert 'page_text_summary' not in page


async def test_frozen_redundant_cycle_stops_early_without_choosing_a_workflow(browser_workspace):
    chunk = {'document_id': 'acme_invoice_1044.pdf', 'chunk_ids': ['acme_invoice_1044.pdf:1:0']}
    responses = [plan(INVOICE, 'browser'), act('inspect_file', document_id=chunk['document_id']),
                 act('read_document_chunks', **chunk), act('browser_open', url='/workspace/finance'),
                 act('browser_type', element_id='@company', text='Acme Corp')]
    for _ in range(4):
        responses.extend([act('read_document_chunks', **chunk), act('browser_open', url='/workspace/finance'),
                          act('browser_type', element_id='@company', text='Acme Corp')])
    provider = FakeProvider(responses)
    before = snapshot_state()
    result = await AgentRunner(provider=provider).execute_task(INVOICE)
    assert result['status'] == 'failed' and 'No progress' in result['error']
    assert result['steps'] == 8 < 20
    assert provider.responses
    assert snapshot_state() == before


async def test_invoice_continuity_preflight_and_observed_503_retry_within_twenty_steps(browser_workspace):
    chunk = {'document_id': 'acme_invoice_1044.pdf', 'chunk_ids': ['acme_invoice_1044.pdf:1:0']}
    responses = [plan(INVOICE, 'browser'), act('inspect_file', document_id=chunk['document_id']),
                 act('read_document_chunks', **chunk), act('browser_open', url='/workspace/finance'),
                 act('browser_type', element_id='@company', text='Acme Corp'),
                 act('read_document_chunks', **chunk), act('browser_open', url='/workspace/finance'),
                 act('browser_type', element_id='@company', text='Acme Corp')]
    responses.extend(act('browser_type', element_id='@' + field, text=value)
                     for field, value in [('invoice_number', 'INV-1044'), ('amount', '84500'), ('due_date', '2026-10-15')])
    responses.extend([act('browser_click', element_id='@submit_invoice'),
                      act('browser_type', element_id='@source_reference', text=chunk['document_id']),
                      act('browser_click', element_id='@submit_invoice'), act('browser_inspect'),
                      act('browser_click', element_id='@submit_invoice'),
                      {'collection': 'invoices', 'company': 'Acme Corp', 'selection': 'latest'}])
    events = []
    before = snapshot_state()
    fault_manager.arm('finance_create_invoice', 1)
    provider = FakeProvider(responses)
    runner = AgentRunner(provider=provider, event_callback=lambda _, kind, payload: events.append((kind, payload)))
    result = await runner.execute_task(INVOICE)
    assert runner.max_steps == 20 and result['status'] == 'completed' and result['steps'] == 15
    assert not provider.responses and result['verification']['verified']
    after_read = context_data(provider.call_history[6])
    remembered = after_read['current_browser_state']
    fields = {field['id']: field for field in remembered['interactive_elements']}
    assert fields['@company']['value'] == 'Acme Corp'
    assert fields['@source_reference']['required'] and fields['@source_reference']['value'] == ''
    assert 'INV-1044' in json.dumps(after_read)
    observations = [payload for kind, payload in events if kind == 'OBSERVATION']
    guarded = next(payload for payload in observations if payload['error_code'] == 'FORM_PRECONDITION_FAILED')
    assert [field['id'] for field in guarded['evidence']['missing_required_fields']] == ['@source_reference']
    transient = next(payload for payload in observations if payload['error_code'] == 'TRANSIENT_503_ERROR')
    assert transient['retriable'] and transient['evidence']['http_status'] == 503
    assert transient['evidence']['commit_state'] == 'not_committed'
    delta = state_delta(before, snapshot_state())
    assert len(delta['invoices']['created']) == 1 and delta['invoices']['created'][0]['amount_minor'] == 8450000
    assert not delta['tickets']['created'] and not delta['accounts']['created']
    assert not any(delta[name][kind] for name in delta for kind in ('updated', 'deleted'))
