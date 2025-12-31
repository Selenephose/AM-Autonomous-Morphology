class SymbolMemory:
    """
    Tracks symbol usage, transitions, and detects reusable / generalized symbols.
    """

    def __init__(self):
        self.usages = {}
        self.transitions = {}
        self.tasks = {}

    def observe(self, symbol, task="default", next_symbol=None):
        self.usages[symbol] = self.usages.get(symbol, 0) + 1

        if symbol not in self.tasks:
            self.tasks[symbol] = set()
        self.tasks[symbol].add(task)

        if symbol not in self.transitions:
            self.transitions[symbol] = set()
        if next_symbol:
            self.transitions[symbol].add(next_symbol)

    def is_general(self, symbol):
        return len(self.tasks.get(symbol, [])) > 1 and self.usages.get(symbol, 0) > 10

    def is_reusable(self, symbol):
        return self.usages.get(symbol, 0) > 7 and len(self.transitions.get(symbol, [])) > 2

    def summary(self):
        return {
            s: {
                "usage": self.usages.get(s, 0),
                "general": self.is_general(s),
                "reusable": self.is_reusable(s),
                "transitions": list(self.transitions.get(s, []))
            } for s in self.usages
        }
