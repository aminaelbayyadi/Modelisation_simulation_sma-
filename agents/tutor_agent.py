from mesa import Agent
import pandas as pd
import os

class TutorAgent(Agent):
    
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.ml_model = None
        self.interventions = 0

    def predict_difficulty(self, student):

        if self.ml_model is None:
            return 0.0

        engagement = student.interactions / (student.charge_travail + 1)

        X = pd.DataFrame([{
            "tutor": self.model.use_tutor,
            "smart_group": self.model.smart_grouping,
            "competence": student.competence,
            "motivation": student.motivation,
            "interactions": student.interactions,
            "charge": student.charge_travail,
            "engagement": engagement
        }])

        probas = self.ml_model.predict_proba(X)

        if probas.shape[1] == 1:
            return 0.0
        else:
            return probas[0][1]

    def intervene(self, student):

        self.interventions += 1

        student.competence += 0.05
        student.motivation += 0.05
        student.satisfaction += 0.08

        student.competence = min(student.competence, 1.0)
        student.motivation = min(student.motivation, 1.0)
        student.satisfaction = min(student.satisfaction, 1.0)

    def step(self):

        # charger modèle une seule fois
        if self.ml_model is None:
            if os.path.exists("data/results.csv"):
                try:
                    from ml.ml_model import train_model
                    self.ml_model = train_model()
                    print(" Modèle IA chargé")
                except:
                    self.ml_model = None

        for student in self.model.students:

            #  NE PAS BLOQUER LES BONS ÉTUDIANTS (important)
            # (on garde seulement une priorité)
            
            if self.ml_model is not None:

                p_drop = self.predict_difficulty(student)

                # 🔴 S4 : intervention rare
                if self.model.auto_learning:
                    if p_drop > 0.8:
                        self.intervene(student)

                # 🟢 S2 : intervention ACTIVE (clé du problème)
                else:
                    if p_drop > 0.5 or student.competence < 0.6:
                        self.intervene(student)

            else:
                # fallback
                if student.competence * student.motivation < 0.5:
                    self.intervene(student)