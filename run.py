## imports config and relevant functions
from feature_gen import ...
from train_models import ...
# from train_models_llm import ... 
import yaml



if __name__ == "__main__":
    with open("../config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    ## train, test = output(feature_gen.py)
    

    ## model, ... = output(train_models.py(train, test))
