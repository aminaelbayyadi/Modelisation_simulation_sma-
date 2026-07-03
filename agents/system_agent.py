from mesa import Agent
from environment.groups import Group
from environment.tasks import Task
from ml.kmeans_model import cluster_students


class SystemAgent(Agent):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.groups = []

    def form_groups(self):
        students = self.model.students.copy()
        self.groups = []
        group_size = 3

        if self.model.smart_grouping:
            # GROUPEMENT INTELLIGENT par K-Means : on classe les etudiants en
            # 3 niveaux (profils), puis on forme les groupes.
            #   "homogene"  -> UN groupe par niveau (faible / moyen / fort)
            #   "heterogene"-> niveaux mixtes (un fort + un moyen + un faible)
            from collections import defaultdict
            labels = cluster_students(students)
            by_level = defaultdict(list)
            for s, lab in zip(students, labels):
                by_level[int(lab)].append(s)
            mode = getattr(self.model, "grouping_mode", "heterogene")

            if mode == "homogene":
                # UN seul groupe par niveau (taille variable selon l'effectif du niveau)
                for lab in sorted(by_level):
                    if by_level[lab]:
                        self.groups.append(Group(by_level[lab]))
            else:  # heterogene : repartir les niveaux pour mixer les groupes
                n_groups = max(1, (len(students) + group_size - 1) // group_size)
                buckets = [[] for _ in range(n_groups)]
                idx = 0
                for lab in sorted(by_level):
                    for s in by_level[lab]:
                        buckets[idx % n_groups].append(s)
                        idx += 1
                for b in buckets:
                    if b:
                        self.groups.append(Group(b))
        else:
            self.random.shuffle(students)
            for i in range(0, len(students), group_size):
                self.groups.append(Group(students[i:i + group_size]))

    def assign_tasks(self):
        for group in self.groups:
            avg_comp = sum(s.competence for s in group.members) / len(group.members)
            if avg_comp < 0.4:
                difficulty = "facile"
            elif avg_comp < 0.7:
                difficulty = "moyenne"
            else:
                difficulty = "difficile" if self.random.random() < 0.3 else "moyenne"
            task = Task(difficulty)
            group.assign_task(task)

    def execute_groups(self):
        for group in self.groups:
            group.work()

    def calculate_metrics(self):
        performances = [s.competence * s.motivation for s in self.model.students]
        if performances and self.model.verbose:
            avg = sum(performances) / len(performances)
            print("Performance moyenne :", avg)

    def step(self):
        # Les groupes et les taches sont desormais formes par le modele
        # AU DEBUT du tour (avant que les etudiants n'agissent), pour que
        # chaque etudiant connaisse ses coequipiers et sa tache commune.
        self.calculate_metrics()