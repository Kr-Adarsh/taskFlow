"""Actual dashboard/SSE rendering against an isolated V2 API run."""
import os
import pytest
from playwright.async_api import async_playwright
from test_real_acceptance import real_workspace as real_workspace
from backend.app.api.runs import set_runner_factory, broadcast_event
from backend.app.agent.provider import FakeProvider
from backend.app.agent_v2.graph import AgentRunner
from backend.app.tools.browser_tools import browser_manager


@pytest.mark.anyio
async def test_dashboard_task_graph_final_report_and_safe_text(real_workspace):
    goal='Read missing source <img src=x onerror=alert(1)>'
    provider=FakeProvider([{'objective':goal,'success_criteria':['Source checked'],'tasks':[{'task_id':'read','goal':goal,'success_criteria':['Source checked'],'verification_capability':'documents'}]}, {'thought':'Missing source','action':'need_clarification','clarification_question':'Which source should I read?'}])
    set_runner_factory(lambda max_steps=20:AgentRunner(provider=provider,max_steps=max_steps,event_callback=broadcast_event))
    try:
        async with async_playwright() as playwright:
            browser=await playwright.chromium.launch(executable_path=os.getenv('TASKFLOW_CHROME_PATH','/usr/bin/google-chrome'),headless=True,args=['--no-sandbox'])
            try:
                page=await browser.new_page()
                await page.goto(browser_manager.base_url+'/dashboard')
                await page.evaluate('setPreset(3)')
                assert 'largest absolute revenue decline' in await page.locator('#objective_input').input_value()
                await page.locator('#objective_input').fill(goal)
                await page.locator('#btn_execute').click()
                await page.wait_for_function("document.getElementById('final_report').textContent.includes('Which source should I read?')")
                assert 'NEEDS_CLARIFICATION' in await page.locator('#task_graph').inner_text()
                assert await page.locator('#task_graph img').count()==0
                assert await page.locator('#verif_status').inner_text()=='NOT VERIFIED'
                assert await page.locator('#btn_execute').is_enabled()
            finally:
                await browser.close()
    finally:
        set_runner_factory(None)
