figure_device_groups = {
    "KBHDP_SOA_1": {
        "Heat": {
            "OPEX": {
                "units": {
                    "fs.costing.total_heat_operating_cost",
                    #   "fs.costing.aggregate_flow_electricity_sold"
                },
            },
        },
        "Electricity": {
            "OPEX": {
                "units": {
                    "fs.costing.aggregate_flow_electricity_purchased",
                    #   "fs.costing.aggregate_flow_electricity_sold"
                },
            },
        },
        "Injection": {
            "OPEX": {
                "units": {"fs.treatment.DWI.unit.costing.variable_operating_cost"},
            },
        },
        "Pumps": {
            "CAPEX": {
                "units": {"fs.treatment.pump.costing.capital_cost"},
            },
        },
        "UF": {
            "CAPEX": {
                "units": {
                    "fs.treatment.UF.unit.costing.capital_cost",
                },
            },
        },
    },
    "KBHDP_RPT_1": {
        "Heat": {
            "OPEX": {
                "units": {
                    "fs.costing.total_heat_operating_cost",
                    #   "fs.costing.aggregate_flow_electricity_sold"
                },
            },
        },
        "Electricity": {
            "OPEX": {
                "units": {
                    "fs.costing.aggregate_flow_electricity_purchased",
                    #   "fs.costing.aggregate_flow_electricity_sold"
                },
            },
        },
        "Injection": {
            "OPEX": {
                "units": {"fs.treatment.DWI.unit.costing.variable_operating_cost"},
            },
        },
        "Pumps": {
            "CAPEX": {
                "units": {"fs.treatment.pump.costing.capital_cost"},
            },
        },
        "PV": {
            "CAPEX": {
                "units": {"fs.energy.pv.costing.capital_cost"},
            },
            "OPEX": {
                "units": {
                    "fs.energy.pv.costing.fixed_operating_cost",
                }
            },
        },
        "EC": {
            "CAPEX": {
                "units": {
                    "fs.treatment.EC.ec.costing.capital_cost",
                },
            },
            "OPEX": {
                "units": {
                    "fs.treatment.EC.ec.costing.fixed_operating_cost",
                    "fs.treatment.costing.aggregate_flow_costs[aluminum]",
                },
            },
        },
        "RO": {
            "CAPEX": {
                "units": {
                    "fs.treatment.RO.stage[1].module.costing.capital_cost",
                }
            },
            "OPEX": {
                "units": {
                    "fs.treatment.RO.stage[1].module.costing.fixed_operating_cost",
                }
            },
        },
        "UF": {
            "CAPEX": {
                "units": {
                    "fs.treatment.UF.unit.costing.capital_cost",
                },
            },
        },
    },
    "KBHDP_RPT_2": {
        "Heat": {
            "OPEX": {
                "units": {
                    "fs.costing.total_heat_operating_cost",
                    #   "fs.costing.aggregate_flow_electricity_sold"
                },
            },
        },
        "Electricity": {
            "OPEX": {
                "units": {
                    "fs.costing.aggregate_flow_electricity_purchased",
                    #   "fs.costing.aggregate_flow_electricity_sold"
                },
            },
        },
        "Injection": {
            "OPEX": {
                "units": {"fs.treatment.DWI.unit.costing.variable_operating_cost"},
            },
        },
        "Pumps": {
            "CAPEX": {
                "units": {"fs.treatment.pump.costing.capital_cost"},
            },
        },
        "PV": {
            "CAPEX": {
                "units": {"fs.energy.pv.costing.capital_cost"},
            },
            "OPEX": {
                "units": {
                    "fs.energy.pv.costing.fixed_operating_cost",
                }
            },
        },
        "EC": {
            "CAPEX": {
                "units": {
                    "fs.treatment.EC.ec.costing.capital_cost",
                },
            },
            "OPEX": {
                "units": {
                    "fs.treatment.EC.ec.costing.fixed_operating_cost",
                    "fs.treatment.costing.aggregate_flow_costs[aluminum]",
                },
            },
        },
        "LTMED": {
            "CAPEX": {
                "units": {
                    "fs.treatment.LTMED.unit.costing.capital_cost",
                },
            },
            "OPEX": {
                "units": {
                    "fs.treatment.LTMED.unit.costing.fixed_operating_cost",
                },
            },
        },
        "UF": {
            "CAPEX": {
                "units": {
                    "fs.treatment.UF.unit.costing.capital_cost",
                },
            },
        },
    },
}
