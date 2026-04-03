class Group:

    def __init__(self, members):
        self.members = members
        self.task = None

    def assign_task(self, task):
        self.task = task

    def work(self):
        results = []

        for student in self.members:
            result = self.task.evaluate(student)
            results.append(result)

        return results