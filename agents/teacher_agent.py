from mesa import Agent
from environment.tasks import Task

class TeacherAgent(Agent):

    # Configuration pédagogique (paramètres modulables)
    SUCCESS_THRESHOLD = 0.6              # seuil global de réussite
    DIFFICULTY_DISTRIBUTION = {           # répartition des difficultés
        "facile": 0.3,                    # 30% des tâches faciles
        "moyenne": 0.5,                   # 50% moyennes
        "difficile": 0.2,                 # 20% difficiles
    }

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        # Consigne pédagogique principale
        self.instructions = "Collaborer pour résoudre les tâches"

        # Indicateurs de classe (mis à jour à chaque step)
        self.completion_rate = 0.0       # taux de complétion des tâches
        self.class_average = 0.0         # performance moyenne de la classe
        self.success_count = 0           # nb d'étudiants au-dessus du seuil

        # Attribution initiale des tâches aux étudiants
        self._assign_initial_tasks()

    # ==============================
    # ATTRIBUTION DES TÂCHES (cas UML : "Attribuer tâches")
    # ==============================
    def _assign_initial_tasks(self):
        """
        Attribue à chaque étudiant une tâche dont la difficulté est tirée
        selon DIFFICULTY_DISTRIBUTION.
        """
        difficulties = list(self.DIFFICULTY_DISTRIBUTION.keys())
        weights = list(self.DIFFICULTY_DISTRIBUTION.values())

        for student in self.model.students:
            # Tirage aléatoire pondéré du niveau de difficulté
            difficulty = self.random.choices(difficulties, weights=weights, k=1)[0]
            student.current_task = Task(difficulty)

    def assign_new_task(self, student):
        """
        Attribue une nouvelle tâche à un étudiant qui vient de terminer
        la sienne (utile pour étendre la simulation).
        """
        difficulties = list(self.DIFFICULTY_DISTRIBUTION.keys())
        weights = list(self.DIFFICULTY_DISTRIBUTION.values())
        difficulty = self.random.choices(difficulties, weights=weights, k=1)[0]
        student.current_task = Task(difficulty)

    # ==============================
    # ÉVALUATION DE LA CLASSE 
    # ==============================
    def evaluate_class(self):
        """
        Évalue la classe entière et met à jour les indicateurs internes.
        Retourne un dict avec les résultats principaux.
        """
        if not self.model.students:
            return {"completion_rate": 0.0, "class_average": 0.0,
                    "success_count": 0}

        # 1) Vérifier la complétion des tâches assignées
        n_completed = 0
        for student in self.model.students:
            task = getattr(student, "current_task", None)
            if task is not None:
                # Une tâche est complétée si la performance dépasse son seuil
                if task.evaluate(student):
                    task.completed = True
                if task.completed:
                    n_completed += 1

        self.completion_rate = n_completed / len(self.model.students)

        # 2) Performance moyenne de la classe
        perfs = [s.competence * s.motivation for s in self.model.students]
        self.class_average = sum(perfs) / len(perfs)

        # 3) Nombre d'étudiants au-dessus du seuil de réussite
        self.success_count = sum(
            1 for p in perfs if p >= self.SUCCESS_THRESHOLD
        )

        return {
            "completion_rate": self.completion_rate,
            "class_average": self.class_average,
            "success_count": self.success_count,
        }

    # ==============================
    # CONSIGNE (cas UML : "Définir consignes")
    # ==============================
    def get_instructions(self):
        """Retourne la consigne pédagogique active (sans print)."""
        return self.instructions

    # ==============================
    # STEP
    # ==============================
    def step(self):
        """
        À chaque tour, l'enseignant évalue la classe (silencieux).
        Aucun print pour ne pas polluer la console pendant les batchs.
        """
        self.evaluate_class()