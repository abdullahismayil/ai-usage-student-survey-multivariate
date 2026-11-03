# Data

The survey data is **not included** in this repository.

It was collected by a five-person student group for a Hacettepe University course project (IST470 Categorical Data Analysis; N = 509 students on the Beytepe campus, 489 after dropping missing usage-frequency answers). It contains individual-level demographics, so it is not redistributed here.

To run the code, put an SPSS file named `survey.sav` in this folder (or pass its path as an argument) with these variables:

| Variable | Coding |
|---|---|
| `Basari` | 1 = GPA 3.00-4.00, 0 = GPA below 3.00 |
| `YZ_Siklik` | Generative-AI use frequency: 1 = Az, 2 = Orta, 3 = Cok |
| `Stres` | Stress level: 1 = Dusuk, 2 = Orta, 3 = Yuksek |
| `Cinsiyet` | 1 = Erkek, 2 = Kadin |
| `Odev`, `Proje`, `Sinav`, `DeepResearch`, `KisiselGelisim`, `SosyalMedya`, `Sohbet`, `SanalAsistan` | Use of AI for eight purposes, 1 = Hic ... 5 = Her zaman |
