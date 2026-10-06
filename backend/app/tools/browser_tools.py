"""Generic browser primitives restricted to one configured workspace origin."""

import json
import os
from pathlib import Path
from urllib.parse import urljoin, urlsplit
from playwright.async_api import async_playwright
from backend.app.tools.base import ToolResult


class BrowserManager:
    def __init__(self, base_url=None):
        self.base_url = (base_url or os.getenv("TASKFLOW_WORKSPACE_URL", "http://127.0.0.1:8000")).rstrip("/")
        self._playwright = self._browser = self._context = self._page = None
        self._elements = {}
        self._blocked = []
        self._responses = []
        self.run_id = None
        self._lease_token = None
        self.last_screenshot_path = None
        self.ui_theme = "light"

    async def set_ui_theme(self, theme):
        """Keep genuine browser captures in the dashboard's presentation theme."""
        if theme not in ("light", "dark"):
            raise ValueError("Invalid browser theme")
        self.ui_theme = theme
        if self._page and not self._page.is_closed():
            await self._page.emulate_media(color_scheme=theme)
            await self._page.evaluate("theme => window.TaskFlowTheme?.set(theme)", theme)

    @property
    def screenshot_root(self):
        return Path(os.getenv("TASKFLOW_SCREENSHOTS_DIR", str(Path(__file__).resolve().parents[3] / "data" / "screenshots"))).resolve()

    async def bind_run(self, run_id, token):
        await self.close()
        self.run_id, self._lease_token = run_id, token
        self.last_screenshot_path = None

    def _is_allowed_url(self, url):
        try:
            candidate = urljoin(self.base_url + "/", url.strip())
            parts, base = urlsplit(candidate), urlsplit(self.base_url)
            origin = lambda p: (p.scheme.lower(), (p.hostname or "").lower(), p.port or (443 if p.scheme == "https" else 80))
            if parts.username is not None or parts.password is not None or parts.scheme not in ("http", "https") or origin(parts) != origin(base):
                return False, "Navigation or request outside the configured workspace is forbidden"
            return True, candidate
        except (ValueError, AttributeError):
            return False, "Invalid workspace URL"

    async def _route_request(self, route):
        allowed, _ = self._is_allowed_url(route.request.url)
        if not allowed:
            self._blocked.append(route.request.url)
            await route.abort("blockedbyclient")
        else:
            # Browser redirects can bypass route callbacks; inspect every HTTP hop ourselves.
            method, body = route.request.method, route.request.post_data
            target = route.request.url
            for _ in range(10):
                response = await route.fetch(url=target, method=method, post_data=body, max_redirects=0, max_retries=0, timeout=10000)
                if response.status not in (301, 302, 303, 307, 308) or not response.headers.get("location"):
                    await route.fulfill(response=response)
                    return
                target = urljoin(target, response.headers["location"])
                if not self._is_allowed_url(target)[0]:
                    self._blocked.append(target)
                    await route.abort("blockedbyclient")
                    return
                if response.status == 303 or (response.status in (301, 302) and method == "POST"):
                    method, body = "GET", None
            self._blocked.append("Workspace redirect limit exceeded")
            await route.abort("blockedbyclient")

    async def start(self):
        if self._page and not self._page.is_closed():
            return
        await self.close()
        self._playwright = await async_playwright().start()
        executable = os.getenv("TASKFLOW_CHROME_PATH", "/usr/bin/google-chrome")
        self._browser = await self._playwright.chromium.launch(executable_path=executable, headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        headers = {"X-TaskFlow-Run": self.run_id, "X-TaskFlow-Lease": self._lease_token} if self.run_id else {}
        self._context = await self._browser.new_context(extra_http_headers=headers, service_workers="block",
                                                       color_scheme=self.ui_theme)
        await self._context.route("**/*", self._route_request)
        self._page = await self._context.new_page()
        self._page.set_default_timeout(10000)
        self._page.set_default_navigation_timeout(15000)
        self._page.on("response", lambda response: self._responses.append(response))
        self._blocked, self._elements = [], {}

    async def get_page(self):
        if not self._page or self._page.is_closed():
            await self.start()
        return self._page

    async def close(self):
        try:
            if self._browser:
                await self._browser.close()
        finally:
            if self._playwright:
                await self._playwright.stop()
            self._playwright = self._browser = self._context = self._page = None
            self._elements = {}

    async def capture_screenshot(self, name_prefix="latest"):
        if not self._page:
            return None
        directory = self.screenshot_root / (self.run_id or "unbound")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "latest.png"
        await self._page.screenshot(path=str(path))
        self.last_screenshot_path = str(path)
        return str(path)

    def _restricted_result(self, previous):
        if len(self._blocked) > previous:
            return ToolResult(ok=False, error="A request outside the workspace was blocked", error_code="RESTRICTED_URL", evidence={"blocked_requests": self._blocked[previous:]})

    @staticmethod
    def _canonical_url(url):
        parts = urlsplit(url)
        return (parts.scheme.lower(), (parts.hostname or '').lower(),
                parts.port or (443 if parts.scheme.lower() == 'https' else 80),
                parts.path or '/', parts.query)

    async def browser_open(self, url):
        allowed, resolved = self._is_allowed_url(url)
        if not allowed:
            return ToolResult(ok=False, error=resolved, error_code="RESTRICTED_URL")
        previous = len(self._blocked)
        try:
            page = await self.get_page()
            previous = len(self._blocked)
            if self._canonical_url(page.url) == self._canonical_url(resolved):
                inspection = await self.browser_inspect()
                blocked = self._restricted_result(previous)
                if blocked:
                    await self.close()
                    return blocked
                if not inspection.ok:
                    return inspection
                return ToolResult(ok=True, error_code="REDUNDANT_ACTION",
                                  data={"current_url": page.url, "title": await page.title(), "inspection": inspection.data,
                                        "already_at_target": True, "no_op": True,
                                        "message": "Already at the requested page; continue from its current state."}, evidence=inspection.evidence)
            response = await page.goto(resolved, wait_until="domcontentloaded")
            blocked = self._restricted_result(previous)
            if blocked:
                await self.close()
                return blocked
            inspection = await self.browser_inspect()
            blocked = self._restricted_result(previous)
            if blocked:
                await self.close()
                return blocked
            if not inspection.ok:
                return inspection
            if response and response.status >= 400:
                return ToolResult(ok=False, error=f"Workspace responded HTTP {response.status}", error_code="HTTP_ERROR", retriable=response.status in (502, 503, 504), evidence=inspection.evidence)
            return ToolResult(ok=True, data={"current_url": page.url, "title": await page.title(), "inspection": inspection.data}, evidence=inspection.evidence)
        except Exception as error:
            blocked = self._restricted_result(previous)
            if blocked:
                await self.close()
                return blocked
            return ToolResult(ok=False, error="Workspace navigation failed", error_code="NAVIGATION_ERROR", evidence={"exception_type":type(error).__name__})

    async def browser_inspect(self):
        try:
            page = await self.get_page()
            if not self._is_allowed_url(page.url)[0]:
                return ToolResult(ok=False, error="Page is outside workspace", error_code="RESTRICTED_URL")
            data = await page.evaluate("""() => {
                const elements = [];
                const used = new Set();
                for (const [index, el] of [...document.querySelectorAll('input:not([type=hidden]),select,textarea,button,a[href]')].entries()) {
                    if (!el.getClientRects().length || getComputedStyle(el).visibility === 'hidden') continue;
                    let id = el.id || el.name || ('element_' + index);
                    if (used.has(id)) id += '_' + index;
                    used.add(id);
                    el.setAttribute('data-taskflow-element', id);
                    elements.push({id: '@' + id, tag: el.tagName.toLowerCase(), type: el.type || '',
                        label: el.labels?.[0]?.textContent.trim() || el.getAttribute('aria-label') || el.textContent.trim() || el.placeholder || '',
                        value: el.value || el.getAttribute('href') || '', required: !!el.required, disabled: el.matches(':disabled'),
                        checked: ['checkbox','radio'].includes(el.type) ? el.checked : undefined,
                        valid: el.willValidate ? el.validity.valid : undefined,
                        form_id: el.form ? (el.form.id || 'form_' + [...document.forms].indexOf(el.form)) : undefined,
                        options: el.options ? [...el.options].map(o => ({value: o.value, label: o.text})) : undefined});
                }
                const feedback = document.getElementById('feedback_message');
                return {interactive_elements: elements, page_text_summary: document.body.innerText.slice(0,4000),
                    source_preview: [...document.querySelectorAll('pre')].map(el => el.innerText).join(' ').slice(0,1600),
                    tables: [...document.querySelectorAll('table')].map(t => [...t.rows].slice(0,20).map(r => [...r.cells].map(c => c.innerText))),
                    feedback: feedback ? {text: feedback.innerText, is_error: feedback.classList.contains('alert-error'), status_code: feedback.dataset.status || '200'} : null};
            }""")
            data.update(url=page.url, title=await page.title())
            self._elements = {el["id"]: el for el in data["interactive_elements"]}
            screenshot = await self.capture_screenshot()
            return ToolResult(ok=True, data=data, evidence={"url": page.url, "screenshot": screenshot})
        except Exception:
            return ToolResult(ok=False, error="Page inspection failed", error_code="INSPECT_FAILED")

    async def _element(self, element_id):
        if element_id not in self._elements:
            raise ValueError("Element was not present in the current observation")
        page = await self.get_page()
        element = page.locator('[data-taskflow-element=' + json.dumps(element_id.removeprefix("@")) + ']')
        if await element.count() != 1 or not await element.is_visible() or await element.is_disabled():
            raise ValueError("Element is stale, ambiguous or unavailable")
        return element

    async def browser_type(self, element_id, text, clear=True):
        try:
            element = await self._element(element_id)
            current_value = await element.input_value()
            value = text if clear else current_value + text
            no_op = current_value == value
            if not no_op:
                await element.fill(value)
            result = await self.browser_inspect()
            if not result.ok:
                return result
            return ToolResult(ok=True, error_code="REDUNDANT_ACTION" if no_op else None,
                              data={"element": element_id, "entered_text": await element.input_value(), "inspection": result.data,
                                    "no_op": no_op, "already_satisfied": no_op}, evidence=result.evidence)
        except Exception:
            return ToolResult(ok=False, error="Observed editable element is unavailable or text is invalid", error_code="TYPE_FAILED")

    async def browser_select(self, element_id, value):
        try:
            element = await self._element(element_id)
            await element.select_option(value=value)
            result = await self.browser_inspect()
            if not result.ok:
                return result
            return ToolResult(ok=True, data={"element": element_id, "selected_value": value, "inspection": result.data}, evidence=result.evidence)
        except Exception:
            return ToolResult(ok=False, error="Observed selection or option is unavailable", error_code="SELECT_FAILED")

    async def browser_click(self, element_id):
        previous = len(self._blocked)
        try:
            element = await self._element(element_id)
            page = await self.get_page()
            old_url = page.url
            form = await element.evaluate("""el => el.form && el.type === 'submit' ? {
                method: el.form.method.toUpperCase(), valid: el.form.checkValidity(),
                required_invalid: [...el.form.elements].filter(x => x.required && !x.disabled && x.willValidate && !x.validity.valid)
                    .map(x => ({id: '@' + (x.getAttribute('data-taskflow-element') || x.id || x.name),
                                label: x.labels?.[0]?.textContent.trim() || x.getAttribute('aria-label') || x.placeholder || '',
                                message: x.validationMessage})),
                invalid: [...el.form.elements].filter(x=>x.validity && !x.validity.valid).map(x=>({id:x.id,message:x.validationMessage}))
            } : null""")
            if form and form['required_invalid']:
                inspection = await self.browser_inspect()
                return ToolResult(ok=False, error="Required enabled form controls must be valid before submission",
                                  error_code="FORM_PRECONDITION_FAILED",
                                  data={"inspection": inspection.data} if inspection.ok else None,
                                  evidence={"missing_required_fields": form['required_invalid'], "commit_state": "not_committed"})
            if form and not form["valid"]:
                return ToolResult(ok=False, error="Required form fields are invalid", error_code="FORM_VALIDATION_ERROR", evidence={"invalid_fields": form["invalid"]})
            self._responses = []
            await element.click()
            await page.wait_for_timeout(250)
            blocked = self._restricted_result(previous)
            if blocked:
                await self.close()
                return blocked
            inspection = await self.browser_inspect()
            blocked = self._restricted_result(previous)
            if blocked:
                await self.close()
                return blocked
            if not inspection.ok:
                return inspection
            evidence = inspection.evidence or {}
            feedback = inspection.data.get("feedback")
            if form and form["method"] == "POST":
                responses = [r for r in self._responses if r.request.method == "POST"]
                evidence["http_status"] = responses[-1].status if responses else None
                if feedback and feedback["is_error"]:
                    status = int(feedback["status_code"])
                    return ToolResult(ok=False, data={"inspection": inspection.data}, error=feedback["text"], error_code="TRANSIENT_503_ERROR" if status == 503 else "FORM_SUBMISSION_ERROR", retriable=status in (502, 503, 504), evidence={**evidence, "commit_state": "not_committed"})
                if feedback and responses and 200 <= responses[-1].status < 300:
                    return ToolResult(ok=True, data={"outcome": "confirmed_mutation", "success_message": feedback["text"], "page_url": page.url, "inspection": inspection.data}, evidence=evidence)
                return ToolResult(ok=False, error="Mutation outcome is uncertain; inspect state before another submission", error_code="UNCERTAIN_OUTCOME", evidence={**evidence, "commit_state": "unknown"})
            if page.url != old_url or (form and form["method"] == "GET"):
                return ToolResult(ok=True, data={"outcome": "confirmed_navigation", "inspection": inspection.data}, evidence=evidence)
            return ToolResult(ok=False, error="Click produced no confirmed navigation or mutation", error_code="UNCERTAIN_OUTCOME", evidence=evidence)
        except Exception:
            return self._restricted_result(previous) or ToolResult(ok=False, error="Click outcome is uncertain; inspect state", error_code="UNCERTAIN_OUTCOME", evidence={"commit_state": "unknown"})

    async def browser_back(self):
        previous = len(self._blocked)
        try:
            page = await self.get_page()
            await page.go_back(wait_until="domcontentloaded")
            return self._restricted_result(previous) or await self.browser_inspect()
        except Exception:
            return self._restricted_result(previous) or ToolResult(ok=False, error="Back navigation failed", error_code="BACK_FAILED")


browser_manager = BrowserManager()
