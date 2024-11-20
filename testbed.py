import copy
import os
import random
import util
import axelrod as axl
from axelrod.action import Action
from axelrod.strategies import WinStayLoseShift
import numpy as np
import matplotlib.pyplot as plt

from ipd_player import ClientShadowPlayer, ResourceAwareMemOnePlayer, RandomIPDPlayer

NUM_GAMES = 20

random.seed(42)

def visualize_scaling():
    # Define parameters
    E_tilde = np.linspace(0, 1.0, 500)  # E_tilde from 0 to 1.0
    E_low = util.ResourceLevel.LOW.value  # Lower threshold
    gamma_values = [4, 8, 12]  # Gamma values

    # Create a function to plot and save the figure
    def plot_and_save(output_folder, file_name):
        plt.figure(figsize=(8, 6))
        for gamma in gamma_values:
            result = util.synergy_threshold_scaling(res_lvl=E_tilde, gamma=gamma)
            plt.plot(E_tilde, result, label=f"γ = {gamma}")
        
        # Highlight areas with different pastel colors
        plt.axvspan(0, E_low, color='lightcoral', alpha=0.2, label="Low Resources")
        plt.axvspan(E_low, 0.5, color='peachpuff', alpha=0.2, label="Moderate Resources")
        plt.axvspan(0.5, 0.75, color='lightgreen', alpha=0.2, label="High Resources")
        plt.axvspan(0.75, 1.0, color='darkseagreen', alpha=0.2, label="Full Resources")

        # Add custom x-axis marks without numerical values except for 0 and 1.0
        x_ticks = [0, E_low, 0.5, 0.75, 1.0]
        x_labels = ["0", r"$E_{\mathrm{Low}}$", r"$E_{\mathrm{Moderate}}$", r"$E_{\mathrm{High}}$", "1.0"]
        plt.xticks(ticks=x_ticks, labels=x_labels)

        # Add labels, legend, and title
        plt.xlabel(r"$\tilde{E}_i$")
        plt.ylabel(r"$f_{\mathrm{res}}(\tilde{E}_i)$")
        plt.title(r"Synergy Threshold Function")
        plt.legend()
        plt.grid(True)

        # Save the plot to the specified folder
        os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist
        save_path = os.path.join(output_folder, file_name)
        plt.savefig(save_path, bbox_inches='tight')  # Save with tight layout
        plt.show()
        print(f"Plot saved to: {save_path}")

    # Example usage
    output_folder = "plots"  # Specify your folder here
    file_name = "highlighted_regions_plot.png"  # Specify your file name here

    # Call the function to plot and save
    plot_and_save(output_folder, file_name)

def test_run(client_player):
    # history list received form server instance
    plays = list()
    coplays = list()
    
    for i in range(NUM_GAMES):
        # init shadow player
        surrogate = ClientShadowPlayer()
        surrogate.set_seed(42)
        
        # reset client strategy
        client_player.reset()
        client_player.set_seed(42)
        
        # inject memory if exists
        if len(plays) > 0:
            surrogate._history.extend(coplays, plays)
            client_player._history.extend(plays, coplays)
        
        # evaluate next move based on given history
        next_action = client_player.strategy(opponent=surrogate)
        plays.append(next_action)
        # assign random action to coplays
        coplays.append(random.choice([Action.C, Action.D]))
        
    print("Finished test run")
    plays_list = "| ".join(play.name for play in plays)
    print("Actions of player one:" + plays_list)
    coplays_list = "| ".join(coplay.name for coplay in coplays)
    print("Actions of player two:" + coplays_list)
    print("----------------")
    

def main():

    # initialize player
    client_player = axl.StochasticWSLS(0)
    #client_player.name = "Random"
    #client_player = axl.GTFT(p=0.0)
    #client_player.name = "T4T"
    client_player.set_seed(42)
    
    res_aware_client_player = ResourceAwareMemOnePlayer(player_instance=copy.deepcopy(client_player),
                                                        resource_scaling_func=util.synergy_threshold_scaling,
                                                        initial_resource_value=util.ResourceLevel.FULL.value)
    
    test_run(res_aware_client_player)
    
    
if __name__ == '__main__':
    visualize_scaling()
    main()