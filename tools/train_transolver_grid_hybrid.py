"""Future grid stage; reuse the frozen full-node trainer and FNO query gradients."""
import argparse
from tools import train_sample_overfitting as trainer
from geobc.overfit_models import backward_case,predict_case
from geobc.overfit_transolver_grid import make_grid_model

def main():
 trainer.make_model=make_grid_model
 trainer.backward_case=lambda method,model,case,guard:backward_case('fno',model,case,guard)
 trainer.predict_case=lambda method,model,case,guard:predict_case('fno',model,case,guard)
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--lock',required=True)
 trainer.run(p.parse_args())

if __name__=='__main__':main()
