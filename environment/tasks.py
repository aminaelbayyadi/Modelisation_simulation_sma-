class Task:

    def __init__(self, difficulty):
        self.difficulty = difficulty
        self.completed = False

    def get_threshold(self):

        # 🔥 SEUILS RÉALISTES
        if self.difficulty == "facile":
            return 0.3

        elif self.difficulty == "moyenne":
            return 0.5

        else:  # difficile
            return 0.7

    def evaluate(self, student):
        performance = student.competence * student.motivation
        return performance > self.get_threshold()