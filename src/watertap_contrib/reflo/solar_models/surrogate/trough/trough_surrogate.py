###############################################################################
# WaterTAP Copyright (c) 2021, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National
# Laboratory, National Renewable Energy Laboratory, and National Energy
# Technology Laboratory (subject to receipt of any required approvals from
# the U.S. Dept. of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#
###############################################################################

import os
import sys
import re

import pandas as pd
import numpy as np

from io import StringIO
import matplotlib.pyplot as plt

from pyomo.environ import (
    Var,
    Param,
    value,
    Expression,
    Constraint,
    Suffix,
    NonNegativeReals,
    units as pyunits,
    check_optimal_termination,
)
from pyomo.common.config import ConfigValue, ListOf

from idaes.core import declare_process_block_class
import idaes.core.util.scaling as iscale
from idaes.core.surrogate.surrogate_block import SurrogateBlock
from idaes.core.surrogate.pysmo_surrogate import PysmoRBFTrainer, PysmoSurrogate
from idaes.core.surrogate.sampling.data_utils import split_training_validation

from watertap_contrib.reflo.core import SolarEnergyBaseData
from idaes.core.solvers.get_solver import get_solver
from idaes.core.util.exceptions import InitializationError

import idaes.logger as idaeslog

_log = idaeslog.getLogger(__name__)

__author__ = "Matthew Boyd, Kurban Sitterley"


@declare_process_block_class("TroughSurrogate")
class TroughSurrogateData(SolarEnergyBaseData):
    """
    Surrogate model for trough.
    """

    CONFIG = SolarEnergyBaseData.CONFIG()
    CONFIG.declare(
        "heat_load_range",
        ConfigValue(
            domain=ListOf(float),
            default=[10, 100],
            description="Range of the Heat Load input",
            doc="""Heat load range [min, max]. Options are: [10, 100] or [100, 500]""",
        ),
    )

    def build(self):
        super().build()

        self.scaling_factor = Suffix(direction=Suffix.EXPORT)
        self._tech_type = "trough"

        self.heat_load = Var(
            initialize=self.config.heat_load_range[0],
            bounds=self.config.heat_load_range,
            units=pyunits.MW,
            doc="Rated plant heat capacity in MWt",
        )

        self.hours_storage = Var(
            initialize=20,
            bounds=[0, 26],
            units=pyunits.hour,
            doc="Rated plant hours of storage",
        )

        self.heat_annual_scaled = Var(
            initialize=1,
            units=pyunits.kWh,
            domain=NonNegativeReals,
            doc="Annual heat generated by trough scaled for surrogate model",
        )

        self.electricity_annual_scaled = Var(
            initialize=1,
            units=pyunits.kWh,
            domain=NonNegativeReals,
            doc="Annual electricity consumed by trough scaled for surrogate model",
        )

        self.heat_annual_scaling = Param(
            initialize=1e-9,
            domain=NonNegativeReals,
            mutable=True,
            doc="Scaling factor of annual heat generated by trough scaled for surrogate model",
        )

        self.electricity_annual_scaling = Param(
            initialize=1e-9,
            domain=NonNegativeReals,
            mutable=True,
            doc="Scaling factor of annual electricity consumed by trough scaled for surrogate model",
        )

        self.heat_annual = Expression(
            expr=self.heat_annual_scaled / self.heat_annual_scaling,
            doc="Annual heat generated by trough in kWh",
        )

        self.electricity_annual = Expression(
            expr=self.electricity_annual_scaled / self.electricity_annual_scaling,
            doc="Annual electricity consumed by trough in kWh",
        )

        stream = StringIO()
        oldstdout = sys.stdout
        sys.stdout = stream

        self.surrogate_inputs = [self.heat_load, self.hours_storage]
        self.surrogate_outputs = [
            self.heat_annual_scaled,
            self.electricity_annual_scaled,
        ]

        self.input_labels = ["heat_load", "hours_storage"]
        self.output_labels = ["heat_annual_scaled", "electricity_annual_scaled"]

        self.dataset_filename = os.path.join(
            os.path.dirname(__file__), "data/trough_data.pkl"
        )
        self.n_samples = 100
        self.training_fraction = 0.8

        heat_load_range = tuple(self.config.heat_load_range)
        self.surrogate_file = os.path.join(
            os.path.dirname(__file__),
            f"trough_surrogate_{int(heat_load_range[0])}_{int(heat_load_range[1])}.json",
        )
        self._create_rbf_surrogate(output_filename=self.surrogate_file)

        self.heat_constraint = Constraint(
            expr=self.heat
            == self.heat_annual
            * pyunits.convert(1 * pyunits.hour, to_units=pyunits.year)
        )

        self.electricity_constraint = Constraint(
            expr=self.electricity
            == self.electricity_annual
            * pyunits.convert(1 * pyunits.hour, to_units=pyunits.year)
        )

        # Revert back to standard output
        sys.stdout = oldstdout

    def calculate_scaling_factors(self):
        if iscale.get_scaling_factor(self.hours_storage) is None:
            sf = iscale.get_scaling_factor(self.hours_storage, default=1)
            iscale.set_scaling_factor(self.hours_storage, sf)

        if iscale.get_scaling_factor(self.heat_load) is None:
            sf = iscale.get_scaling_factor(self.heat_load, default=1, warning=True)
            iscale.set_scaling_factor(self.heat_load, sf)

        if iscale.get_scaling_factor(self.heat_annual_scaled) is None:
            sf = iscale.get_scaling_factor(
                self.heat_annual_scaled, default=1, warning=True
            )
            iscale.set_scaling_factor(self.heat_annual_scaled, sf)

        if iscale.get_scaling_factor(self.heat) is None:
            sf = iscale.get_scaling_factor(self.heat, default=1, warning=True)
            iscale.set_scaling_factor(self.heat, sf)

        if iscale.get_scaling_factor(self.electricity_annual_scaled) is None:
            sf = iscale.get_scaling_factor(
                self.electricity_annual_scaled, default=1, warning=True
            )
            iscale.set_scaling_factor(self.electricity_annual_scaled, sf)

        if iscale.get_scaling_factor(self.electricity) is None:
            sf = iscale.get_scaling_factor(self.electricity, default=1e-3, warning=True)
            iscale.set_scaling_factor(self.electricity, sf)

    def initialize_build(
        blk,
        outlvl=idaeslog.NOTSET,
        solver=None,
        optarg=None,
    ):
        """
        General wrapper for initialization routines

        Keyword Arguments:
            outlvl : sets output level of initialization routine
            optarg : solver options dictionary object (default=None)
            solver : str indicating which solver to use during
                     initialization (default = None)

        Returns: None
        """
        init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="unit")
        solve_log = idaeslog.getSolveLogger(blk.name, outlvl, tag="unit")

        # Initialize surrogate
        data = pd.DataFrame(
            {
                "heat_load": [value(blk.heat_load)],
                "hours_storage": [value(blk.hours_storage)],
            }
        )
        test_output = blk.surrogate.evaluate_surrogate(data)
        blk.heat_annual_scaled.set_value(test_output.heat_annual_scaled.values[0])
        blk.electricity_annual_scaled.set_value(
            test_output.electricity_annual_scaled.values[0]
        )
        blk.heat.set_value(value(blk.heat_annual) / 8766)
        blk.electricity.set_value(value(blk.electricity_annual) / 8766)

        # Solve unit
        opt = get_solver(solver, optarg)
        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(blk, tee=slc.tee)

        init_log.info_high(f"Initialization Step 2 {idaeslog.condition(res)}")

        if not check_optimal_termination(res):
            raise InitializationError(f"Unit model {blk.name} failed to initialize")

        init_log.info("Initialization Complete: {}".format(idaeslog.condition(res)))

    def _create_rbf_surrogate(self, data_training=None, output_filename=None):
        if data_training is None:
            self._get_surrogate_data()
        else:
            self.data_training = data_training

        # Capture long output
        stream = StringIO()
        oldstdout = sys.stdout
        sys.stdout = stream

        # Create PySMO trainer object
        self.trainer = PysmoRBFTrainer(
            input_labels=self.input_labels,
            output_labels=self.output_labels,
            training_dataframe=self.data_training,
        )

        # Set PySMO options
        self.trainer.config.basis_function = "gaussian"  # default = gaussian
        self.trainer.config.solution_method = "algebraic"  # default = algebraic
        self.trainer.config.regularization = True  # default = True

        # Train surrogate
        self.rbf_train = self.trainer.train_surrogate()

        # Remove autogenerated 'solution.pickle' file
        try:
            os.remove("solution.pickle")
        except FileNotFoundError:
            pass
        except Exception as e:
            raise e

        # Create callable surrogate object
        xmin, xmax = [self.config.heat_load_range[0], 0], [
            self.config.heat_load_range[1],
            26,
        ]
        self.input_bounds = {
            self.input_labels[i]: (xmin[i], xmax[i])
            for i in range(len(self.input_labels))
        }
        self.surrogate = PysmoSurrogate(
            self.rbf_train, self.input_labels, self.output_labels, self.input_bounds
        )

        # Save model to JSON
        if output_filename is not None:
            model = self.surrogate.save_to_file(output_filename, overwrite=True)

        # Revert back to standard output
        sys.stdout = oldstdout

        self.surrogate_blk = SurrogateBlock(concrete=True)
        self.surrogate_blk.build_model(
            self.surrogate,
            input_vars=self.surrogate_inputs,
            output_vars=self.surrogate_outputs,
        )

    def _get_surrogate_data(self, return_data=False):
        self.pickle_df = pd.read_pickle(self.dataset_filename)
        heat_load_range = self.config.heat_load_range
        self.pickle_df = self.pickle_df[
            (self.pickle_df["heat_load"] >= heat_load_range[0])
            & (self.pickle_df["heat_load"] <= heat_load_range[1])
        ]
        self.data = self.pickle_df.sample(n=self.n_samples)
        self.data_training, self.data_validation = split_training_validation(
            self.data, self.training_fraction, seed=len(self.data)
        )

        # Scaling based on the subset of data
        scaling_factors = [self.heat_annual_scaling, self.electricity_annual_scaling]

        for n, label in enumerate(self.output_labels):
            label_unscaled = label.split("_scaled")[0]
            output_max = self.data_training[label_unscaled].max()
            self.data_training.loc[:, label] = (
                self.data_training[label_unscaled] / output_max
            )
            scaling_factors[n].set_value(1 / output_max)

        if return_data:
            return self.data_training, self.data_validation

    def _plot_training_validation(
        self,
        data_training=None,
        data_validation=None,
        surrogate=None,
        surrogate_filename="trough_surrogate.json",
    ):
        if data_training is None and data_validation is None:
            data_training = self.data_training
            data_validation = self.data_validation

        if surrogate is None and surrogate_filename is not None:
            surr_file = os.path.join(os.path.dirname(__file__), surrogate_filename)
            surrogate = PysmoSurrogate.load_from_file(surr_file)
        elif surrogate is None and surrogate_filename is None:
            raise Exception
        else:
            surrogate = self.surrogate

        for output_label in self.output_labels:
            # Output fit metrics and create parity and residual plots
            print(
                "\n{label}: \n\tR-squared: {r2} \n\tRMSE: {rmse}".format(
                    label=output_label.replace("_", " ").title(),
                    r2=surrogate._trained._data[output_label].model.R2,
                    rmse=surrogate._trained._data[output_label].model.rmse,
                )
            )
            training_output = surrogate.evaluate_surrogate(
                data_training[self.input_labels]
            )
            label = re.sub(
                "[^a-zA-Z0-9 \n\.]", " ", output_label.title()
            )  # keep alphanumeric chars and make title case
            self._parity_residual_plots(
                true_values=np.array(data_training[output_label]),
                modeled_values=np.array(training_output[output_label]),
                label=label + " - Training",
            )

            # Validate model using validation data
            validation_output = surrogate.evaluate_surrogate(
                data_validation[self.input_labels]
            )
            self._parity_residual_plots(
                true_values=np.array(data_validation[output_label]),
                modeled_values=np.array(validation_output[output_label]),
                label=label + " - Validation",
            )

    def _parity_residual_plots(
        self,
        true_values,
        modeled_values,
        label=None,
        figx=9,
        figy=5,
        axis_fontsize=12,
        title_fontsize=15,
    ):
        fig1 = plt.figure(figsize=(figx, figy), tight_layout=True)
        if label is not None:
            fig1.suptitle(label, fontsize=title_fontsize)
        ax = fig1.add_subplot(121)
        ax.plot(true_values, true_values, "-")
        ax.plot(true_values, modeled_values, "o")
        ax.set_xlabel(r"True data", fontsize=axis_fontsize)
        ax.set_ylabel(r"Surrogate values", fontsize=axis_fontsize)
        ax.set_title(r"Parity plot", fontsize=axis_fontsize)

        ax2 = fig1.add_subplot(122)
        ax2.plot(
            true_values,
            true_values - modeled_values,
            "s",
            mfc="w",
            mec="m",
            ms=6,
        )
        ax2.axhline(y=0, xmin=0, xmax=1)
        ax2.set_xlabel(r"True data", fontsize=axis_fontsize)
        ax2.set_ylabel(r"Residuals", fontsize=axis_fontsize)
        ax2.set_title(r"Residual plot", fontsize=axis_fontsize)

        plt.show()