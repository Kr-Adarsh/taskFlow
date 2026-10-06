import json
from backend.app.capabilities.documents import chunks, read_document_chunks, inspect_file
from backend.app.agent_v2.context import ContextMemory, decision_prompt
from backend.app.agent_v2.state import GraphState, Subtask
from backend.app.workspace.seed import reset_demo_env
from backend.app.tools.base import ToolResult


def test_chunks_retain_provenance_and_exact_selection():
    reset_demo_env()
    source=chunks('acme_invoice_1044.pdf')
    assert source and all({'document_id','page','section','chunk_id','text'}<=chunk.keys() for chunk in source)
    result=read_document_chunks('acme_invoice_1044.pdf',chunk_ids=[source[0]['chunk_id']])
    assert result.ok and result.data['chunks'][0]==source[0]
    assert not read_document_chunks('acme_invoice_1044.pdf',chunk_ids=['invented']).ok
    assert not inspect_file('../../.env').ok


def test_context_does_not_repeat_transcript_or_previous_dom():
    memory=ContextMemory()
    memory.update('read_document_chunks',{},ToolResult(ok=True,data={'chunks':[{'document_id':'doc','chunk_id':'doc:1:0','text':'Fact: value','page':'1','section':'first'}]}))
    task=Subtask(task_id='read',goal='Read facts',success_criteria=['Facts sourced'],verification_capability='documents')
    state=GraphState(run_id='context',objective='Read facts',tasks=[task],current_task_id='read',observation={'data':{'inspection':{'url':'/x','interactive_elements':[{'id':'field'}],'tables':[],'page_text_summary':'old DOM '*2000}}})
    messages=decision_prompt(state,task,memory,[])
    text=json.dumps(messages)
    assert 'Fact: value' in text and 'old DOM' not in text
    assert 'untrusted_task_data' in text


def test_source_preview_and_crm_facts_survive_navigation():
    memory=ContextMemory()
    for url,tables,preview in [('/workspace/documents',[[['Title'],['Complaint']]],''),('/workspace/documents?view=source.txt',[[['Title'],['Complaint']]],'Customer: Example\nIssue: outage'),('/workspace/crm',[[['Name','Tier'],['Example','Enterprise']]],''),('/workspace/support',[[['Customer','Priority']]],'')]:
        memory.update('browser_open',{'url':url},ToolResult(ok=True,data={'inspection':{'url':url,'tables':tables,'source_preview':preview}}))
    context=memory.relevant()
    assert next(item['facts']['text'] for item in context['established_evidence'] if item['kind']=='browser_preview').startswith('Customer: Example')
    assert next(item['facts']['tables'] for item in context['established_evidence'] if item['source']=='/workspace/crm')[0][1]==['Example','Enterprise']
    assert len(memory.pages)==3
