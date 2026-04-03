class Task:

    def __init__(self, difficulty):
        self.difficulty = difficulty
        self.completed = False

    def get_threshold(self):
        if self.difficulty == "facile":
            return 0.4
        elif self.difficulty == "moyenne":
            return 0.6
        else:
            return 0.8

    def evaluate(self, student):
        performance = student.competence * student.motivation
        return performance > self.get_threshold()