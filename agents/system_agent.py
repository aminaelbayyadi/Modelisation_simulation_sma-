from mesa import Agent
from environment.groups import Group
from environment.tasks import Task
from ml.kmeans_model import cluster_students


class SystemAgent(Agent):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.groups = []

    # ==============================
    # FORMATION DES GROUPES
    # ==============================
    def form_groups(self):
        students = self.model.students.copy()
        self.groups = []
        group_size = 3

        if self.model.smart_grouping:

            labels = cluster_students(students)

            clusters = {}
            for student, label in zip(students, labels):
                clusters.setdefault(label, []).append(student)

            # mélanger tous les étudiants
            mixed = []
            for c in clusters.values():
                mixed.extend(c)

            self.random.shuffle(mixed)

            for i in range(0, len(mixed), group_size):
                self.groups.append(Group(mixed[i:i+group_size]))

        else:
            self.random.shuffle(students)

            for i in range(0, len(students), group_size):
                self.groups.append(Group(students[i:i+group_size]))

    # ==============================
    # ATTRIBUTION DES TÂCHES
    # ==============================
    def assign_tasks(self):

        for group in self.groups:

            avg_comp = sum(s.competence for s in group.members) / len(group.members)

            if avg_comp < 0.4:
                difficulty = "facile"

            elif avg_comp < 0.7:
                difficulty = "moyenne"

            else:
                #  parfois difficile (pas toujours)
                if self.random.random() < 0.3:
                    difficulty = "difficile"
                else:
                    difficulty = "moyenne"

            task = Task(difficulty)
            group.assign_task(task)

    # ==============================
    # EXÉCUTION
    # ==============================
    def execute_groups(self):
        for group in self.groups:
            group.work()

    # ==============================
    # MÉTRIQUES
    # ==============================
    def calculate_metrics(self):

        performances = [
            s.competence * s.motivation
            for s in self.model.students
        ]

        if performances:
            avg = sum(performances) / len(performances)
            print("Performance moyenne :", avg)

    # ==============================
    # STEP
    # ==============================
    def step(self):

        #  IMPORTANT : groupes à chaque tour
        self.form_groups()

        self.assign_tasks()
        self.execute_groups()
        self.calculate_metrics()