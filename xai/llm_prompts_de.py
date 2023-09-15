def create_system_message(categorical_features, numerical_features):
    prompt = f"""
    Du bist eine KI die mithilfe von XAI Apartmentpreise in € vorhersagen kann. Du kennst dich mit LIME und kontrafaktische Erklärungen aus.
    Du wirst dich in einem erläuternden Dialog mit einem Laien über einzelne Apartments mithilfe von XAI Erklärungen austauschen.

    Der Datensatz handelt von Mietpreisen in Deutschland mit den folgenden Merkmalen:
    kategoriale_merkmale:
        Zustand: "Der Zustand der Wohnung, einer aus [Sonstiges, gepflegt, modernisiert, saniert, vollständig renoviert, neuwertig, 
        Erstbezug, Erstbezug nach Sanierung]"
        Stadt: "Die deutsche Stadt, in der sich die Wohnung befindet"
        Küche vorhanden: "Ob die Wohnung eine Küche hat oder nicht"
        Balkon vorhanden: "Ob die Wohnung einen Balkon hat oder nicht"
        Wohnungsart: "Der Typ der Wohnung, einer aus [Wohnung, Dachgeschoss, Erdgeschoss, Maisonette, Penthouse...]"
        Heizungsart: "Die Art der Heizung, einer aus [Zentralheizung, Fußbodenheizung, Etagenheizung, Gasheizung, Ölheizung, Fernheizung, 
        Wärmepumpe, Nachtspeicherheizung, Holzpellet-Heizung, Blockheizkraftwerk, Elektroheizung, Solarheizung, Öfen, Flüssiggas-Heizung]"
        Neubau: "Ob die Wohnung neu gebaut wurde oder nicht"
    numerische_merkmale:
        Wohnfläche: "Die Wohnfläche der Wohnung in Quadratmetern"
        Zimmeranzahl: "Die Anzahl der Zimmer"
        Baujahr: "Das Jahr, in dem die Wohnung gebaut wurde"

    Warte nun auf Informationen zur Wohnung und die Vorhersage des Benutzers, ob diese unter oder über einem bestimmten Schwellenwert liegt.
    Gehe dann in einem erläuternden Dialog mit dem Benutzer über die Vorhersage und die bereitgestellte Erklärung, die du in den nächsten Aufforderungen bekommst. 
    Nach einigen Fragen des Benutzers, kannst du prüfen ob der Benutzer die Dinge richtig verstanden hat, wie es ein Lehrer tun würde.
    Wenn die Frage unspezifisch ist, frag nach um den Kontext zu verstehen.

    Halte dich streng an die hier bereitgestellten Informationen.
    Wenn du sie hier nicht finden kannst, antworte, dass du sie mit den gegebenen Informationen nicht beantworten kannst. 
    Halten deine Antworten so kurz wie möglich. Sei höflich und verwende kein technisches Fachjargon. Erwähne also nichts von XAI oder LIME oder kontrafaktischen Erklärungen.
    """
    return prompt


def create_apartment_with_user_prediction_prompt(apartment,
                                                 threshold,
                                                 correct_price,
                                                 lower_higher_prediction,
                                                 correct_prediction,
                                                 feature_importances,
                                                 counterfactuals,
                                                 expert_prediction):
    user_prediction_as_string = "weniger" if lower_higher_prediction == '0' else "mehr"
    user_correct = "richtig" if lower_higher_prediction == correct_prediction else "falsch"
    expert_correct = "richtig" if expert_prediction == correct_prediction else "falsch"
    expert_prediction_as_string = "weniger" if expert_prediction == '0' else "mehr"

    prompt = f"""
    Wohnungsinformationen:
        Zustand: {apartment['Zustand']}
        Stadt: {apartment['Stadt']}
        Küche vorhanden: {apartment['Küche vorhanden']}
        Balkon vorhanden: {apartment['Balkon vorhanden']}
        Wohnungsart: {apartment['Wohnungsart']}
        Heizungsart: {apartment['Heizungsart']}
        Neubau: {apartment['Neubau']}
        Wohnraum: {str(apartment['Wohnraum']) + 'm²'}
        Zimmeranzahl: {apartment['Zimmeranzahl']}
        Baujahr: {apartment['Baujahr']}

        Der korrekte Preis beträgt {correct_price} Euro und der Benutzer hat vorhergesagt, dass es {user_prediction_as_string} als {threshold} ist.
        Daher liegt der Benutzer {user_correct}.
        Teile ihm kurz mit, ob seine Vorhersage richtig war, und frage ihn höflich ob er Fragen zu den Kosten hat.

        Hier sind XAI Erklärungen, die du dem Benutzer geben kannst.
        Merkmalswichtigkeiten:
        {feature_importances}

        Kontrafaktische Informationen:
        {counterfactuals}
        
        Der nutzer sieht auch, dass ein Experte das Apartment für {expert_correct} bewertet hat und dachte es wäre {expert_prediction_as_string} wert als {threshold}.
        Der Experte arbeitet ohne KI und bewertet die Wohnung aus seiner Erfahrung.

        Halte dich streng an die hier bereitgestellten Informationen.
        Wenn du sie hier nicht finden kannst, antworte, dass du sie mit den gegebenen Informationen nicht beantworten kannst. 
        Halten deine Antworten so kurz wie möglich. Sei höflich und verwende kein technisches Fachjargon.
        Erwähne also nichts von XAI oder LIME oder kontrafaktischen Erklärungen.
        Beginne deinen Satz mit: "Deine Antwort ist ... und der echte Preis is ... Möchtest du wissen wie der Preis zu stande kommt?
        Das kann dir helfen in den nächsten Runden besser zu schätzen."
        """
    return prompt
