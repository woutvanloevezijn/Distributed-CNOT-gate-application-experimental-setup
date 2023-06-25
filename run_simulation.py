import numpy as np

from application import ControllerProgram, TargetProgram
from squidasm.run.stack.config import StackConfig, StackNetworkConfig, LinkConfig, DepolariseLinkConfig, HeraldedLinkConfig, GenericQDeviceConfig, NVQDeviceConfig
from squidasm.run.stack.run import run
import multiprocessing as mp
import os
import math
import matplotlib.pyplot as plt


# PERFORMANCE METRIC PARAMETERS
# set inputs of the performance metric to the z-basis basis (0) or x-basis (1)
input_type = 0

# experiment data, the syntax per experiment is ((<c_theta>,<c_phi>),(<t_theta>,<t_phi>))
if input_type == 0:
    inputs = [((0, 0), (0, 0)), ((0, 0), (math.pi, 0)), ((math.pi, 0), (0, 0)), ((math.pi, 0), (math.pi, 0))]
    outputs = [((0, 0), (0, 0)), ((0, 0), (math.pi, 0)), ((math.pi, 0), (math.pi, 0)), ((math.pi, 0), (0, 0))]
    experiment_names = ["00", "01", "10", "11"]
else:
    inputs = [((0.5*math.pi, 0),(0.5*math.pi, 0)),((0.5*math.pi, 0), (0.5*math.pi, math.pi)),((0.5*math.pi, math.pi),(0.5*math.pi, 0)),((0.5*math.pi, math.pi), (0.5*math.pi, math.pi))]
    outputs = [((0.5*math.pi, 0),(0.5*math.pi, 0)),((0.5*math.pi, math.pi), (0.5*math.pi, math.pi)),((0.5*math.pi, math.pi),(0.5*math.pi, 0)),((0.5*math.pi, 0), (0.5*math.pi, math.pi))]
    experiment_names = ["++","+-","-+","--"]

# select performance metric to error rate of the controller node (0), target node (1) or combined (2)
bit_select = 2


# DATA POINT PARAMETERS
# total amount of runs per data point, n = simulation_iterations * epr_rounds
simulation_iterations = 20
epr_rounds = 50

# values of the hardware parameters that will be plotted
start_value = 0.0
end_value = 1.0
step_size = (end_value - start_value)/10

# set the tested device parameter on the control node (0) or target node (1)
stack = 0

# link type: "depolarise" or "heralded"
link_type = "heralded"


# SYSTEM PARAMETERS
# amout of parallel processses
processes=12
# amout in which the data points per plot are divided between processes
nr_of_sections = int(processes/len(inputs))


# CHART PARAMETERS
x_axis_name = "Fidelity"
y_axis_name = "Error rate"
# title = "Error rate to " + x_axis_name + " in "+ ("z-basis" if input_type == 0 else "x-basis") + ", " + str(simulation_iterations * epr_rounds) + " runs"
title = "Error rate to " + x_axis_name + " of " + ("c node" if stack == 0 else "t node") + " in "+ ("z-basis" if input_type == 0 else "x-basis") + ", " + str(simulation_iterations * epr_rounds) + " runs"
colors = ['k', 'c', 'm', 'y']
save_data = 1
dir_path = "/home/wout/Documents/TUDelft/CSE3000 RP/experiments/distributed_cnot/"



# import network configuration from file
cfg = StackNetworkConfig.from_file("config.yaml")

# set the accurate link configuration
if link_type == "depolarise": link_config = DepolariseLinkConfig.from_file("depolarise_link_config.yaml")
if link_type == "heralded": link_config = HeraldedLinkConfig.from_file("heralded_link_config.yaml")
link = LinkConfig(stack1="Controller", stack2="Target", typ=link_type, cfg=link_config)

# replace link from YAML file with new depolarise link
cfg.links = [link]


# runs experiments on the same performance metric for data points in the selected section in parallel to other processes
# returns the experiment index, x_values, mean and standard error of the datapoints for the given section
def experiment_process(experiment, section):

    print(f"Running experiment: {experiment_names[experiment]}, Section: {section}, PID: {os.getpid()}")

    # set a parameter, the number of epr rounds for the programs, the input type and bit select
    controller_program = ControllerProgram(num_epr_rounds=epr_rounds, input_theta=inputs[experiment][0][0], input_phi=inputs[experiment][0][1], output_theta=outputs[experiment][0][0], output_phi=outputs[experiment][0][1], bit_select=bit_select)
    target_program = TargetProgram(num_epr_rounds=epr_rounds, input_theta=inputs[experiment][1][0], input_phi=inputs[experiment][1][1], output_theta=outputs[experiment][1][0], output_phi=outputs[experiment][1][1])

    # generate values for the hardware parameter of interest
    x_values = np.arange(start_value, end_value, step=step_size)
    # slice the section to compute
    x_values = x_values[int((section/nr_of_sections)*len(x_values)):int(((section+1)/nr_of_sections)*len(x_values))]

    data_points = []
    for x_value in x_values:

        # SPECIFY HARDWARE PARAMETER
        link_config.fidelity = x_value
        # cfg.stacks[stack].qdevice_cfg['electron_single_qubit_depolar_prob'] = x_value

        # run the simulation
        # return values from run method are the results per node
        results_controller, results_target = run(config=cfg,
                                        programs={"Controller": controller_program, "Target": target_program},
                                        num_times=simulation_iterations)

        # select results either of controller, target or both (also controller)
        results = results_controller
        if bit_select == 1:
            results = results_target
        
        # estimator for the SD of the sampled mean
        sd = math.sqrt(sum([results[i]["var_error"]/epr_rounds for i in range(simulation_iterations)]))/simulation_iterations
        avgs_final_state = [results[i]["avg_error"] for i in range(simulation_iterations)]
        mean = sum(avgs_final_state) / simulation_iterations


        data_point = {"experiment": experiment, "x_value": x_value, "mean": mean, "sd": sd}
        data_points.append(data_point)
        print(f"Result - PID: {os.getpid()}, {data_point}")

    return data_points

# functions for sorting
def keyXValue(e):
    return e["x_value"]
def keyExperiment(e):
    return e["experiment"]

# print settings
print(title)
print(f"Error rate of {('target' if bit_select == 1 else 'controller' if bit_select == 0 else 'combined')} Qubit")
print(f"Using {simulation_iterations * epr_rounds} runs per data point")

if __name__ == '__main__':
    # use multithreading settings for UNIX systems
    context = mp.get_context('fork')
    
    # create a pool of processors which run differen experiments in multiple sections
    results = []
    with context.Pool(processes=processes) as pool:
        # run all four inputs
        results = [[pool.apply_async(experiment_process, (experiment, section,)) for section in range(nr_of_sections)] for experiment in range(len(inputs))]
        results = [[result.get() for result in results[experiment]] for experiment in range(len(inputs))]
        pool.close()

    # flatten results
    new_results = []
    for experiment in results:
        for list in experiment:
            for result in list: 
                new_results.append(result)
    results = new_results

    # sort results on the x-axis and input
    results.sort(key=keyXValue)
    results.sort(key=keyExperiment)
    # plot all results
    for experiment in range(len(inputs)):
        x_values = np.arange(start_value, end_value, step=step_size)
        means = [result["mean"] for result in results if result["experiment"] == experiment]
        sds = [result["sd"] for result in results if result["experiment"] == experiment]

        # minimise the square error on a 2nd degree polynomial
        polyfit = np.polyfit(x_values, means, 2)

        model = np.poly1d(polyfit)
        line = np.linspace(start_value, results[len(results)-1]["x_value"], 100)
        plt.errorbar(x_values, means, sds, fmt ='o', capsize=4, color=colors[experiment])
        plt.plot(line, model(line), color=colors[experiment], label = str(experiment_names[experiment]))

        # calculate maximum slope in the given data range
        sensitivity = max(abs(2*polyfit[0]*start_value+polyfit[1]),abs(2*polyfit[0]*results[len(results)-1]["x_value"]+[polyfit[1]]))
        print(f"Experiment: {experiment}, Sensitivity: {sensitivity}")

# print data points
print(f"DATA POINTS: {title}")
for result in results:
    print(f"{experiment_names[result['experiment']]}, {result['x_value']}, {result['mean']}, {result['sd']}")

# write data points to file
if save_data:
    with open(dir_path + title + ".txt", 'w') as f:
        f.write(f"DATA POINTS: {title}\n")
        for result in results:
            f.write(f"{experiment_names[result['experiment']]}, {result['x_value']}, {result['mean']}, {result['sd']}\n")

# format graph
plt.title(title)
plt.xlabel(x_axis_name)
plt.ylabel(y_axis_name)
plt.legend()
if save_data: plt.savefig(dir_path + title + ".png")
plt.show()