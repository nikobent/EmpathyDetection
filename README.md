# EmpathyDetection
In this repository I include parts of my work for my Master's thesis with the title "Text-based empathy detection on social media" <br>

My thesis can be divided into two parts: <br>
1. Create a transformers based system able to predict the Empathetic Concern and Personal Distress scores of users' comments <br>
2. Predict with the already created model the EC and PD scores of users comments on Twitter and Reddit <br>

For the creation of the models I used the dataset created by Buechel:  https://github.com/wwbp/empathic_reactions <br>
The files experiment.py and helper.py were used to apply different Transformer models on this dataset. I developed these models using PyTorch and their implementations from HuggingFace.<br>

In addition there exist python notebooks for statistical analysis
