"""Bound repeated semantic cycles without choosing the agent's next action."""
from collections import deque
import hashlib
import json
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def canonical_url(value):
    parsed = urlsplit(value)
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path,
                      urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True))), ''))


def compact_page(page):
    fields = ('id', 'tag', 'type', 'value', 'checked', 'required', 'disabled', 'valid', 'form_id', 'selected_option')
    controls = sorted([{key: element[key] for key in fields if key in element}
                       for element in page.get('interactive_elements', [])[:40]], key=lambda item: item['id'])
    noise = {'timestamp', 'created_at', 'updated_at', 'request_id'}
    tables = []
    for table in page.get('tables', [])[:8]:
        if not table:
            continue
        columns = [index for index, name in enumerate(table[0][:32])
                   if str(name).lower().replace(' ', '_') not in noise]
        rows = [[[table[0][index], row[index]] for index in columns if index < len(row)]
                for row in table[1:41]]
        tables.append(sorted([sorted(row, key=lambda cell: str(cell[0])) for row in rows], key=digest))
    return {'url': canonical_url(page.get('current_url') or page.get('url') or ''),
            'controls': controls, 'tables': sorted(tables, key=digest)}


def evidence_versions(memory):
    versions = {}
    for name, source in memory.get('sources', {}).items():
        for chunk_id, chunk in source.get('chunks', {}).items():
            versions['chunk:' + name + ':' + chunk_id] = digest(chunk.get('text', ''))
    for url, excerpt in memory.get('source_excerpts', {}).items():
        versions['source:' + canonical_url(url)] = digest(excerpt.get('text', ''))
    for name, profile in memory.get('dataset_profiles', {}).items():
        versions['dataset:' + name] = digest({key: profile[key] for key in
            ('sha256', 'shape', 'columns', 'dtypes') if key in profile})
    return versions


class SemanticOscillationGuard:
    def __init__(self):
        self.history = deque(maxlen=12)
        self.pages = {}
        self.evidence = {}
        self.evidence_marker = None
        self.business_state = None
        self.mutation_revision = 0
        self.current_page = None
        self.pending = None
        self.warned = False

    def _business_changed(self, state):
        marker = digest(state)
        changed = self.business_state is not None and self.business_state != marker
        self.business_state = marker
        if changed:
            self.mutation_revision += 1
            self.pending = None
            self.warned = False
        return changed

    @staticmethod
    def action_key(tool, args):
        arguments = dict(args)
        if tool == 'browser_open' and isinstance(arguments.get('url'), str):
            arguments['url'] = canonical_url(arguments['url'])
        return digest({'tool': tool, 'args': arguments})

    def observe(self, tool, args, result, memory, business_state):
        changed = self._business_changed(business_state)
        if result.retriable:
            self.history.clear()
            self.pending = None
            self.warned = False
            return
        if not result.ok:
            return
        evidence = evidence_versions(memory)
        evidence_marker = digest(evidence)
        new_evidence = self.evidence_marker != evidence_marker
        self.evidence_marker = evidence_marker
        self.evidence.update(evidence)
        while len(self.evidence) > 64:
            del self.evidence[next(iter(self.evidence))]
        url = memory.get('current_browser_url')
        page = memory.get('pages', {}).get(url)
        if not page:
            if changed or new_evidence:
                self.pending = None
                self.warned = False
            return
        state = compact_page(page)
        page_hash = digest(state)
        new_page_state = self.pages.get(state['url']) != page_hash
        self.pages[state['url']] = page_hash
        while len(self.pages) > 8:
            del self.pages[next(iter(self.pages))]
        self.current_page = state
        novel = changed or new_evidence or new_page_state or tool == 'execute_python'
        fingerprint = digest({'page': state, 'evidence': evidence,
                              'mutation_revision': self.mutation_revision})
        if novel:
            self.pending = None
            self.warned = False
        self.history.append({'fingerprint': fingerprint, 'page': state['url'], 'novel': novel,
                             'action_key': self.action_key(tool, args), 'tool': tool,
                             'args': {key: value for key, value in args.items() if key != 'code'}})
        entries = list(self.history)
        for length in range(2, 5):
            if len(entries) < length * 2:
                continue
            first, second = entries[-length * 2:-length], entries[-length:]
            pattern = [entry['fingerprint'] for entry in second]
            if (len(set(pattern)) > 1 and pattern == [entry['fingerprint'] for entry in first]
                and not any(entry['novel'] for entry in second)):
                self.pending = {'length': length, 'entries': second}
                break

    def check_action(self, tool, args, memory, business_state):
        if self._business_changed(business_state) or not self.pending:
            return None
        pattern = self.pending['entries']
        equivalent = self.action_key(tool, args) in {entry['action_key'] for entry in pattern}
        # A different navigation tool can still revisit the same known page.
        target = args.get('url') if tool == 'browser_open' else None
        if tool == 'browser_click' and self.current_page:
            control = next((item for item in self.current_page['controls']
                            if item['id'] == args.get('element_id') and item.get('tag') == 'a'), None)
            target = control.get('value') if control else None
        if target and self.current_page:
            equivalent |= canonical_url(urljoin(self.current_page['url'], target)) in {entry['page'] for entry in pattern}
        if not equivalent:
            return None
        fatal = self.warned
        self.warned = True
        available = sorted(list(memory.get('sources', {})) + list(memory.get('source_excerpts', {})))
        return {'fatal': fatal, 'cycle_length': self.pending['length'],
                'pattern': [{'state': entry['fingerprint'][:12], 'page': entry['page'],
                             'tool': entry['tool'], 'args': entry['args']} for entry in self.pending['entries']],
                'available_evidence': available[:12], 'remembered_pages': sorted(memory.get('pages', {})),
                'message': 'Previously observed evidence remains available in working memory. '
                           'Choose a different action that adds evidence or changes state; this equivalent cycle was not executed.'}
