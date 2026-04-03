from mesa import Agent
import pandas as pd
import os

class TutorAgent(Agent):
    
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.ml_model = None  # modèle IA

    def predict_difficulty(self, student):
        """
        Prédire si l'étudiant est en difficulté
        """

        if self.ml_model is None:
            return False

        #  Feature engineering
        engagement = student.interactions / (student.charge_travail + 1)

        #  DataFrame 
        X = pd.DataFrame([{
            "tutor": self.model.use_tutor,
            "smart_group": self.model.smart_grouping,
            "competence": student.competence,
            "motivation": student.motivation,
            "interactions": student.interactions,
            "charge": student.charge_travail,
            "engagement": engagement
        }])

        prediction = self.ml_model.predict(X)

        return prediction[0]

    def intervene(self, student):
        """
        Intervention intelligente du tuteur
        """

        student.competence += 0.1
        student.motivation += 0.1
        student.satisfaction += 0.1

        #  NORMALISATION 
        student.competence = max(0, min(student.competence, 1.0))
        student.motivation = max(0, min(student.motivation, 1.0))
        student.satisfaction = max(0, min(student.satisfaction, 1.0))

    def step(self):
        """
        Action du tuteur à chaque itération
        """

        #  Charger modèle UNE SEULE FOIS
        if self.ml_model is None:
            if os.path.exists("data/results.csv"):
                try:
                    from ml.ml_model import train_model
                    self.ml_model = train_model()
                    print(" Modèle IA chargé")
                except Exception as e:
                    print(" Erreur chargement IA :", e)
                    self.ml_model = None

        #  Parcourir les étudiants
        for student in self.model.students:

            #  IA ou fallback
            if self.ml_model is not None:
                is_difficult = self.predict_difficulty(student)
            else:
                is_difficult = student.competence * student.motivation < 0.5

            #  Intervention si nécessaire
            if is_difficult:
                self.intervene(student)