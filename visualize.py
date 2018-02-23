import pandas as pd
import utils
import matplotlib.pyplot as plt

def plot_results(params, data):
    position_error = data["position error"]
    pose_error = data["pose error"]
    total_error = data["loss"]
    plt.plot(position_error, label="position error")
    plt.plot(pose_error, label="pose error")
    plt.plot(total_error, label="total error")
    plt.xlabel("# Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(params.viz_output_file)

if __name__ == "__main__":
    params = utils.Params("./params.json")
    results = pd.read_csv(params.viz_file)
    plot_results(params, results)
