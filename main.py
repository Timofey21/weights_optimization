import os
import subprocess
import json
import numpy as np


CONFIDENCE_WEIGHTS_JSON = '/var/www/guarddog-research-testing/guarddog/confidence_weights.json'
SEVERITY_WEIGHTS_JSON = '/var/www/guarddog-research-testing/guarddog/severity_weights.json'

EXPERIMENTS_DIR = '/var/www/experiments'

def read_json(json_file):
    with open(json_file, 'r') as f:
        result = json.load(f)
    return result

def update_json(json_file, weights):
    with open(json_file, 'r') as f:
        config = json.load(f)

    config.update(weights)

    with open(json_file, 'w') as file:
        json.dump(config, file, indent=4)

# mean squared error function 
def mse(labeled, predicted):
    return np.mean((labeled - predicted) ** 2)

# run guarddog tool to analyze pacakges from epxeriments directory
def run_guarddog(file):

    target_directory = "/var/www/guarddog-research-testing"
    os.chdir(target_directory)


    activate = f"source {os.path.join(target_directory, 'venv', 'bin', 'activate')}"
    python_executable = os.path.join(target_directory, "venv", "bin", "python")


    script_to_run = "-m guarddog npm scan"
    packages_directory = "/var/www/experiments/"

    directory = packages_directory + file

    command = f"{activate} && {python_executable} {script_to_run} {directory}"
    process = subprocess.Popen(command, shell=True, executable='/bin/bash', 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    stdout, _ = process.communicate()
    
    output_lines = stdout.strip().split('\n')
    rules_list = output_lines[-3] 
    confidence = output_lines[-2] 
    severity = output_lines[-1]

    rules_list = [rules.strip() for rules in rules_list.split(',')]

    confidence = float(confidence)
    severity = float(severity)

    process.terminate()
    process.wait()

    return confidence, severity, rules_list
 

def gradient(file, labeled):

    eps = 0.3

    confidence_weights = read_json(CONFIDENCE_WEIGHTS_JSON)
    severity_weights = read_json(SEVERITY_WEIGHTS_JSON)


    grad_conf = {key: 0 for key in confidence_weights}
    grad_severity = {key: 0 for key in severity_weights}

    list = run_guarddog(file)
    rules_list = list[2]

    for rule in confidence_weights:
        if rule not in rules_list:
            grad_conf[rule] = 0
            grad_severity[rule] = 0
            continue

        confidence_weights[rule] += eps
        severity_weights[rule] += eps
        update_json(CONFIDENCE_WEIGHTS_JSON, confidence_weights)
        update_json(SEVERITY_WEIGHTS_JSON, severity_weights)
        pred_plus = run_guarddog(file)
        loss_plus_conf = mse(labeled[0], pred_plus[0])
        loss_plus_severity = mse(labeled[1], pred_plus[1])

        confidence_weights[rule] -= 2 * eps
        severity_weights[rule] -= 2 * eps
        update_json(CONFIDENCE_WEIGHTS_JSON, confidence_weights)
        update_json(SEVERITY_WEIGHTS_JSON, severity_weights)
        pred_minus = run_guarddog(file)
        loss_minus_conf = mse(labeled[0], pred_minus[0])
        loss_minus_severity = mse(labeled[1], pred_minus[1])

        confidence_weights[rule] += eps
        severity_weights[rule] += eps
        update_json(CONFIDENCE_WEIGHTS_JSON, confidence_weights)
        update_json(SEVERITY_WEIGHTS_JSON, severity_weights)

        grad_conf[rule] = (loss_plus_conf - loss_minus_conf) / 2 * eps
        grad_severity[rule] = (loss_plus_severity - loss_minus_severity) / 2 * eps

    return grad_conf, grad_severity

def gradient_descent(file, labeled):

    learning_rate = 0.1
    iterations = 100

    confidence_weights = read_json(CONFIDENCE_WEIGHTS_JSON)
    severity_weights = read_json(SEVERITY_WEIGHTS_JSON)

    for i in range(iterations):
        confidence_grad, severity_grad = gradient(file, labeled)
        
        for key in confidence_weights:
            confidence_weights[key] -= learning_rate * confidence_grad[key]           

        for key in severity_weights:
            severity_weights[key] -= learning_rate * severity_grad[key]
        
        update_json(CONFIDENCE_WEIGHTS_JSON, confidence_weights)
        update_json(SEVERITY_WEIGHTS_JSON, severity_weights)


        predicted = run_guarddog(file)
        loss_confidence = mse(labeled[0], predicted[0])
        loss_severity = mse(labeled[1], predicted[1])


        print(f"Itr: {i}. Confidence score loss func: {loss_confidence}, Severity score loss func: {loss_severity}")

  

if __name__ == "__main__":

    labeled_result = read_json("file_results.json")
    file_names = [file for file in os.listdir(EXPERIMENTS_DIR) if os.path.isfile(os.path.join(EXPERIMENTS_DIR, file))]


    for file in file_names:
        conf_result = float(labeled_result[file]['confidence'])
        severity_result = float(labeled_result[file]['severity'])
        results = conf_result, severity_result

        gradient_descent(file, results)
