import os
import pandas as pd
from mesa import Agent


class TutorAgent(Agent):

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.interventions = 0
        # Le tuteur utilise le Random Forest pre-entraine fourni par le modele.
        self.ml_model = getattr(model, "ml_model", None)

    def predict_difficulty(self, student):
        """Predit le risque de decrochage de l'etudiant via le Random Forest.
        Memes features que l'entrainement (sans competence/motivation -> pas de leakage).
        """
        if self.ml_model is None:
            # Repli heuristique si aucun modele n'a ete fourni
            return max(0.0, 1.0 - student.competence * student.motivation)
        engagement = student.interactions / (student.charge_travail + 1)
        X = pd.DataFrame([{
            "tutor": self.model.use_tutor,
            "smart_group": self.model.smart_grouping,
            "interactions": student.interactions,
            "charge": student.charge_travail,
            "engagement": engagement,
        }])
        try:
            probas = self.ml_model.predict_proba(X)
            if probas.shape[1] == 1:
                return 0.0
            return float(probas[0][1])
        except Exception:
            return max(0.0, 1.0 - student.competence * student.motivation)

    def intervene(self, student):
        self.interventions += 1
        student.competence += 0.05
        student.motivation += 0.05
        student.satisfaction += 0.08
        student.competence = min(student.competence, 1.0)
        student.motivation = min(student.motivation, 1.0)
        student.satisfaction = min(student.satisfaction, 1.0)

    def _should_intervene(self, student, p_drop):
        if self.model.auto_learning:
            return p_drop > 0.8
        # Politique IDENTIQUE avec ou sans groupes intelligents : le tuteur
        # intervient de la meme facon partout. On ne confond donc pas l'effet
        # du groupement avec une difference d'intensite d'aide du tuteur.
        return p_drop > 0.5 or student.competence < 0.6

    def step(self):
        students = self.model.students
        # Prediction du risque pour TOUS les etudiants en une seule fois (rapide).
        if self.ml_model is not None:
            X = pd.DataFrame([{
                "tutor": self.model.use_tutor,
                "smart_group": self.model.smart_grouping,
                "interactions": s.interactions,
                "charge": s.charge_travail,
                "engagement": s.interactions / (s.charge_travail + 1),
            } for s in students])
            try:
                probas = self.ml_model.predict_proba(X)
                risks = probas[:, 1] if probas.shape[1] > 1 else [0.0] * len(students)
            except Exception:
                risks = [max(0.0, 1.0 - s.competence * s.motivation) for s in students]
        else:
            risks = [max(0.0, 1.0 - s.competence * s.motivation) for s in students]

        for student, p_drop in zip(students, risks):
            if self._should_intervene(student, p_drop):
                self.intervene(student)