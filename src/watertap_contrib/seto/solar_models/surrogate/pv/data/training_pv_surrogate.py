import os
from os.path import join, dirname
import sys
import re
import numpy as np
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
from pyomo.environ import ConcreteModel, SolverFactory, value, Var, Objective, maximize
from pyomo.common.timing import TicTocTimer
from idaes.core.surrogate.sampling.data_utils import split_training_validation
from idaes.core.surrogate.pysmo_surrogate import PysmoRBFTrainer, PysmoSurrogate
from idaes.core.surrogate.surrogate_block import SurrogateBlock
from idaes.core import FlowsheetBlock

def create_rbf_surrogate(
    training_dataframe, input_labels, output_labels, output_filename=None
):
    # Capture long output
    stream = StringIO()
    oldstdout = sys.stdout
    sys.stdout = stream

    # Create PySMO trainer object
    trainer = PysmoRBFTrainer(
        input_labels=input_labels,
        output_labels=output_labels,
        training_dataframe=training_dataframe,
    )

    # Set PySMO options
    trainer.config.basis_function = "gaussian"  # default = gaussian
    trainer.config.solution_method = "algebraic"  # default = algebraic
    trainer.config.regularization = True  # default = True

    # Train surrogate
    rbf_train = trainer.train_surrogate()

    # Remove autogenerated 'solution.pickle' file
    try:
        os.remove("solution.pickle")
    except FileNotFoundError:
        pass
    except Exception as e:
        raise e

    # Create callable surrogate object
    xmin, xmax = [100, 0], [1000, 26]
    input_bounds = {
        input_labels[i]: (xmin[i], xmax[i]) for i in range(len(input_labels))
    }
    rbf_surr = PysmoSurrogate(rbf_train, input_labels, output_labels, input_bounds)

    # Save model to JSON
    if output_filename is not None:
        model = rbf_surr.save_to_file(output_filename, overwrite=True)

    # Revert back to standard output
    sys.stdout = oldstdout

    # Display first 50 lines and last 50 lines of output
    # celloutput = stream.getvalue().split('\n')
    # for line in celloutput[:50]:
    #     print(line)
    # print('.')
    # print('.')
    # print('.')
    # for line in celloutput[-50:]:
    #     print(line)

    return rbf_surr