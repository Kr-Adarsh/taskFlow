from backend.app.tools.registry import ToolRegistry


def schema(**properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties)}


class CapabilityRegistry(ToolRegistry):
    def __init__(self):
        super().__init__()
        self.categories = {}

    def add(self, category, name, description, parameters, func):
        self.register(name, description, parameters, func)
        self.categories[name] = category


def build_registry():
    from backend.app.capabilities.browser import register_browser
    from backend.app.capabilities.documents import register_documents
    from backend.app.capabilities.python import register_python
    registry = CapabilityRegistry()
    register_browser(registry)
    register_documents(registry)
    register_python(registry)
    return registry
