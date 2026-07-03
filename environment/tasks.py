class Task:

    _id_counter = 0

    REWARDS = {
        "facile":    {"competence": 0.003, "motivation": 0.005},
        "moyenne":   {"competence": 0.007, "motivation": 0.007},
        "difficile": {"competence": 0.012, "motivation": 0.010},
    }

    THRESHOLDS = {
        "facile":    0.30,
        "moyenne":   0.50,
        "difficile": 0.70,
    }

    def __init__(self, difficulty):
        Task._id_counter += 1
        self.task_id = Task._id_counter

        if difficulty not in self.THRESHOLDS:
            difficulty = "moyenne"
        self.difficulty = difficulty
        self.completed = False

    def get_threshold(self):
        return self.THRESHOLDS[self.difficulty]

    def evaluate(self, student):
        performance = student.competence * student.motivation
        return performance >= self.get_threshold()

    def attempt(self, student):
        
        if self.evaluate(student):
            self.completed = True
            reward = self.REWARDS[self.difficulty]
            student.competence = min(1.0, student.competence + reward["competence"])
            student.motivation = min(1.0, student.motivation + reward["motivation"])
            return True
        return False

    def __repr__(self):
        status = " completee" if self.completed else " non completee"
        return f"Task#{self.task_id}({self.difficulty}, {status})"