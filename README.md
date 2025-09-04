# Analiza i predikcija rezultata testova (Student performance)

Ovaj projekat vrši analizu rezultata veštine matematike, čitanja i pisanja koristeći dataset "Students Performance" preuzet sa Kaggle sajta.

## Kako pokrenuti

1) Treniranje modela i čuvanje najboljeg modela:
   `python train.py`
2) Pokretanje streamlit lokalne aplikacije:
   `streamlit run streamlit_app.py`

## Šta se nalazi u projektu
- **Analiza (eda.ipynb)**: Jedan Jupyter fajl sa grafikonima: raspodele poena, boxplot, korelacije i par poređenja po grupama (ručak, priprema za test, roditeljsko obrazovanje…). Ideja je da se vidi ko je u proseku iznad/ispod i koliko i da se izvrši odgovarajuća analiza podataka koji dovode do nekih logičkih zaključaka.

- **Treniranje modela (train.py)**  
  Skripta koja pripremi podatke i uporedi par modela (neuronska mreža – MLP, plus još 2 jednostavnija modela). Ispiše rezultate po predmetima i sačuva najbolji model u fajl `best_model.joblib`.

- **Aplikacija (streamlit_app.py)**  
  Mali interfejs gde izabereš pol, ručak, pripremu, obrazovanje roditelja i grupu; klikneš „Predict” i dobiješ procenu **math / reading / writing** poena.