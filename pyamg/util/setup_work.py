"""Logical setup work for one independently benchmarked configuration."""


class SetupWork(list):
    """Count each dependency once, even if another configuration cached it.

    Dependencies already paid for by hierarchy construction are excluded.
    Plain list counters elsewhere continue to count physical construction work.
    """

    def __init__(self, excluded=()):
        super().__init__([0, 0])
        self.seen = set(excluded)

    def charge(self, matrix, dependency, numerical, graph=0):
        key = self.key(matrix, dependency)
        if key not in self.seen:
            self.seen.add(key)
            self[0] += numerical
            self[1] += graph

    @staticmethod
    def key(matrix, dependency):
        # Tokens avoid id reuse without retaining temporary sparse matrices.
        if not hasattr(matrix, '_setup_work_token'):
            matrix._setup_work_token = object()
        return (matrix._setup_work_token, dependency)
