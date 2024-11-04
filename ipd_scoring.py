import os
import matplotlib.pyplot as plt
import util
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


# payoff-matrix configuration
R = 3 # CC
S = 0 # CD
T = 5 # DC
P = 1 # DD

def get_ipd_score(a1, a2):
    s1 = s2 = 0
    # depending on the payoff-matrix, assign score - hard code for now
    if a1 and a2:
        s1 = R
        s2 = R
    elif a1 and not a2:
        s1 = S
        s2 = T
    elif not a1 and a2:
        s1 = T
        s2 = S
    elif not a1 and not a2:
        s1 = P
        s2 = P
    return s1, s2
        
        
def update_scoreboard(
        ipd_scoreboard_dict, 
        match_id: int, 
        c1_res_tuple: tuple, 
        c2_res_tuple: tuple, 
        server_round: int
        ):
        """
    Updates the scoreboard with the results of a match round between two clients.

    Parameters:
    - match_id (int): Unique identifier for the match.
    - c1_res_tuple (tuple): A tuple containing client 1's data in the form 
                            (client_id, play, payoff, ipd_strategy, res_level).
    - c2_res_tuple (tuple): A tuple containing client 2's data in the form 
                            (client_id, play, payoff, ipd_strategy, res_level).
    - server_round (int): The current round number in the server game loop.

    Each client's result is appended as a tuple:
    (server_round, match_id, opponent_id, play, coplay, payoff, ipd_strategy, res_level).
    """
        # Ensure each client has an entry in the scoreboard dictionary
        ipd_scoreboard_dict.setdefault(c1_res_tuple[0], [])
        ipd_scoreboard_dict.setdefault(c2_res_tuple[0], [])
        
        # Update data for client 1
        ipd_scoreboard_dict[c1_res_tuple[0]].append(
            (server_round, match_id, c2_res_tuple[0], c1_res_tuple[1], c2_res_tuple[1], c1_res_tuple[2], c1_res_tuple[3], c1_res_tuple[4])
        )
        # Update data for client 2
        ipd_scoreboard_dict[c2_res_tuple[0]].append(
            (server_round, match_id, c1_res_tuple[0], c2_res_tuple[1], c1_res_tuple[1], c2_res_tuple[2], c2_res_tuple[3], c2_res_tuple[4])
        )


def get_ranked_payoffs(ipd_scoreboard_dict):
    """
    Calculates the total payoffs for each client and returns a ranked list.

    Parameters:
        - scoreboard_dict: the scoreboard
        
    Returns:
    - List of tuples, where each tuple is in the format (client_id, total_payoff), 
    sorted by total_payoff in descending order.
    """
    # Dictionary to hold the total payoff for each client
    total_payoffs = {}

    # Calculate total payoffs for each client
    for client_id, rounds in ipd_scoreboard_dict.items():
        # Sum up the payoffs for this client across all rounds
        total_payoffs[client_id] = sum(round[5] for round in rounds)

    # Sort clients by total payoffs in descending order and return as a list of tuples
    ranked_payoffs = sorted(total_payoffs.items(), key=lambda x: x[1], reverse=True)

    return ranked_payoffs


def print_ranked_payoffs(ipd_scoreboard_dict):
    """
    Calculates and prints the total payoffs, strategy, and resource level for each client in a ranked table format.
    """
    # Dictionary to hold the total payoff, strategy, and resource level for each client
    client_info = {}

    # Calculate total payoffs, and get strategy and resource level for each client
    for client_id, rounds in ipd_scoreboard_dict.items():
        # Sum up the payoffs for this client across all rounds
        total_payoff = sum(round[5] for round in rounds)  # payoff is at index 5

        # Extract strategy and resource level from the first entry (since they are constant)
        strategy = rounds[0][6]  # ipd_strategy is at index 6
        resource_level = rounds[0][7]  # res_level is at index 7

        # Store the information
        client_info[client_id] = (total_payoff, strategy, resource_level)

    # Sort clients by total payoffs in descending order
    ranked_clients = sorted(client_info.items(), key=lambda x: x[1][0], reverse=True)

    # Print the header for the ranking table
    print(f"{'Rank':<5} {'Client ID':<10} {'Total Payoff':<15} {'Strategy':<20} {'Resource Level':<15}")
    print("-" * 70)

    # Print each client's rank, ID, total payoff, strategy, and resource level
    for rank, (client_id, (total_payoff, strategy, resource_level)) in enumerate(ranked_clients, start=1):
        print(f"{rank:<5} {client_id:<10} {total_payoff:<15} {strategy:<20} {resource_level:<15.2f}")


def plot_payoffs_over_rounds(ipd_scoreboard_dict):
    """
    Creates a line plot of each client's cumulative payoffs over the number of server rounds,
    labeling each client by their strategy name.
    """
    # Dictionary to store cumulative payoffs over rounds for each client
    cumulative_payoffs = {}

    # Loop through each client to calculate cumulative payoffs by round
    for client_id, rounds in ipd_scoreboard_dict.items():
        cumulative_payoff = 0
        rounds_list = []
        payoffs_list = []
        
        for round_data in rounds:
            # round_data format: (server_round, match_id, opponent_id, play, coplay, payoff, ipd_strategy, res_level)
            server_round = round_data[0]
            payoff = round_data[5]  # payoff is at index 5

            # Update cumulative payoff
            cumulative_payoff += payoff
            rounds_list.append(server_round)
            payoffs_list.append(cumulative_payoff)
        
        # Get the strategy for the client from the first round entry
        strategy_name = rounds[0][6]  # ipd_strategy is at index 6
        cumulative_payoffs[strategy_name] = (rounds_list, payoffs_list)

    # Plot each client's cumulative payoff over rounds with strategy labels
    plt.figure(figsize=(10, 6))
    for strategy_name, (rounds_list, payoffs_list) in cumulative_payoffs.items():
        plt.plot(rounds_list, payoffs_list, label=strategy_name)

    # Add titles and labels
    plt.title("Cumulative Payoffs Over Server Rounds")
    plt.xlabel("Server Round")
    plt.ylabel("Cumulative Payoff")
    plt.legend(title="Strategy")
    plt.grid(True)
    plt.show()
    
    
    
def format_ranked_payoffs_for_logging(ipd_scoreboard_dict):
    """
    Formats the total payoffs, strategy, resource level, number of games, and average payoff
    for each client in a ranked table format, as a string suitable for logging.

    Returns:
    - A formatted string with total payoffs ranked, including strategy, resource level,
      number of games, and average payoff.
    """
    # Dictionary to hold the total payoff, strategy, resource level, number of games, and average payoff for each client
    client_info = {}

    # Calculate total payoffs, and get strategy, resource level, and number of games for each client
    for client_id, rounds in ipd_scoreboard_dict.items():
        # Sum up the payoffs for this client across all rounds they participated in
        total_payoff = sum(round[5] for round in rounds)  # payoff is at index 5

        # Extract strategy and resource level from the first entry (assuming they are constant)
        strategy = rounds[0][6]  # ipd_strategy is at index 6
        resource_level = (util.ResourceLevel.from_float(rounds[0][7])).to_string()  # res_level is at index 7

        # Calculate the number of games (rounds this client actually played)
        num_games = len(rounds)
        
        # Calculate the average payoff, handling cases where num_games is zero
        average_payoff = total_payoff / num_games if num_games > 0 else 0

        # Store the information
        client_info[client_id] = (total_payoff, strategy, resource_level, num_games, average_payoff)

    # Sort clients by total payoffs in descending order
    ranked_clients = sorted(client_info.items(), key=lambda x: x[1][0], reverse=True)

    # Build the formatted string for logging
    output = []
    output.append(" ")
    output.append(f"{'Rank':<5} {'Client ID':<10} {'Total Payoff':<15} {'Strategy':<35} {'Resource Level':<15} {'Games':<10} {'Avg Payoff':<15}")
    output.append("-" * 105)

    # Append each client's rank, ID, total payoff, strategy, resource level, number of games, and average payoff
    for rank, (client_id, (total_payoff, strategy, resource_level, num_games, average_payoff)) in enumerate(ranked_clients, start=1):
        output.append(f"{rank:<5} {client_id:<10} {total_payoff:<15} {strategy:<35} {resource_level:<15} {num_games:<10} {average_payoff:<15.2f}")

    # Join all lines into a single formatted string
    formatted_output = "\n".join(output)
    
    return formatted_output


def plot_unique_strategy_confusion_matrix(ipd_scoreboard_dict):
    """
    Plots a confusion matrix showing the frequency of interactions between unique clients,
    labeled by their strategy and resource level, ensuring only one unique interaction per round.
    """
    # Dictionary to store counts of interactions between unique client labels
    interaction_counts = defaultdict(int)

    # Track processed pairs for each round to ensure unique interactions
    processed_pairs = set()  # This will store tuples of (server_round, label_pair)

    # Iterate through each client to count unique interactions
    for client_id, rounds in ipd_scoreboard_dict.items():
        for round_data in rounds:
            # round_data format: (server_round, match_id, opponent_id, play, coplay, payoff, ipd_strategy, res_level)
            server_round = round_data[0]
            match_id = round_data[1]
            client_label = f"{round_data[6]} | {round_data[7]}"  # Client's strategy and resource level
            opponent_id = round_data[2]

            # Ensure the opponent exists and retrieve opponent data for the same round
            if opponent_id in ipd_scoreboard_dict:
                opponent_round = next((r for r in ipd_scoreboard_dict[opponent_id] if r[1] == match_id), None)
                if opponent_round:
                    opponent_label = f"{opponent_round[6]} | {opponent_round[7]}"

                    # Sort labels alphabetically for a unique pair key
                    label_pair = tuple(sorted([client_label, opponent_label]))

                    # Use (server_round, label_pair) as the unique key for each interaction per round
                    unique_key = (server_round, label_pair)
                    if unique_key not in processed_pairs:
                        processed_pairs.add(unique_key)
                        interaction_counts[label_pair] += 1

    # Extract unique labels and create a matrix
    unique_labels = sorted(set(label for pair in interaction_counts.keys() for label in pair))
    matrix = pd.DataFrame(0, index=unique_labels, columns=unique_labels)

    # Fill the confusion matrix based on the unique interaction counts
    for (label_1, label_2), count in interaction_counts.items():
        matrix.at[label_1, label_2] = count
        if label_1 != label_2:
            matrix.at[label_2, label_1] = count  # Symmetric matrix

    # Plot the confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Unique Strategy and Resource Level Confusion Matrix")
    plt.xlabel("Client (Strategy | Resource Level)")
    plt.ylabel("Opponent (Strategy | Resource Level)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.show()
    
    

def plot_strategy_scores_matrix(ipd_scoreboard_dict):
    """
    Plots a confusion matrix showing the total scores of each client against others,
    formatted as '(sum_score1, sum_score2) (number of interactions)'.
    """
    # Dictionary to store accumulated scores and interaction counts between unique strategy pairs
    # Key: (client_label, opponent_label)
    # Value: [sum_client_scores, sum_opponent_scores, interaction_count]
    interaction_data = {}

    # Iterate through each client to collect scores for unique interactions
    for client_id, rounds in ipd_scoreboard_dict.items():
        for round_data in rounds:
            # round_data format:
            # (server_round, match_id, opponent_id, play, coplay, payoff, ipd_strategy, res_level)
            match_id = round_data[1]
            client_label = f"{round_data[6]} | {round_data[7]}"  # Client's strategy and resource level
            opponent_id = round_data[2]
            client_payoff = round_data[5]  # Client's payoff for this round

            # Ensure the opponent exists and retrieve opponent data for the same match
            if opponent_id in ipd_scoreboard_dict:
                # Find the opponent's data within the same match
                opponent_round = next((r for r in ipd_scoreboard_dict[opponent_id] if r[1] == match_id), None)
                if opponent_round:
                    opponent_label = f"{opponent_round[6]} | {opponent_round[7]}"
                    opponent_payoff = opponent_round[5]  # Opponent's payoff for this round

                    # Create a label pair
                    label_pair = (client_label, opponent_label)

                    # Initialize or update the interaction data
                    if label_pair not in interaction_data:
                        interaction_data[label_pair] = [client_payoff, opponent_payoff, 1]
                    else:
                        interaction_data[label_pair][0] += client_payoff
                        interaction_data[label_pair][1] += opponent_payoff
                        interaction_data[label_pair][2] += 1

    # Extract unique labels
    unique_labels = sorted(set(label for pair in interaction_data.keys() for label in pair))

    # Create a DataFrame to store the formatted scores
    matrix = pd.DataFrame("", index=unique_labels, columns=unique_labels)

    # Fill the matrix with the formatted scores
    for (label_1, label_2), (sum_score1, sum_score2, interaction_count) in interaction_data.items():
        matrix.at[label_1, label_2] = f"({sum_score1}, {sum_score2}) ({interaction_count})"

    # Plot the confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix.isin(['']).astype(int), annot=matrix.values, fmt='', cmap="Blues", cbar=False)
    plt.title("Strategy Scores Matrix with Total Scores and Interaction Counts")
    plt.xlabel("Opponent (Strategy | Resource Level)")
    plt.ylabel("Client (Strategy | Resource Level)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.show()
    
    
def plot_strategy_total_scores_over_rounds(ipd_scoreboard_dict):
    """
    Plots the cumulative total scores obtained by each strategy over the server rounds.
    """
    # Collect all unique server rounds and strategies
    all_rounds = set()
    strategies = set()
    data_list = []

    # Gather data from ipd_scoreboard_dict
    for client_id, rounds in ipd_scoreboard_dict.items():
        for round_data in rounds:
            server_round = round_data[0]
            strategy_label = f"{round_data[6]} | {round_data[7]}"  # Strategy name | Resource level
            payoff = round_data[5]  # Payoff

            all_rounds.add(server_round)
            strategies.add(strategy_label)
            data_list.append((server_round, strategy_label, payoff))

    # Sort the server rounds
    sorted_rounds = sorted(all_rounds)

    # Initialize cumulative scores and totals for each strategy
    cumulative_scores_per_strategy = {strategy: [] for strategy in strategies}
    cumulative_totals = {strategy: 0 for strategy in strategies}

    # Group data by server round
    data_by_round = defaultdict(list)
    for server_round, strategy_label, payoff in data_list:
        data_by_round[server_round].append((strategy_label, payoff))

    # Iterate over each server round in order
    for server_round in sorted_rounds:
        # Append current cumulative totals to the lists
        for strategy in strategies:
            cumulative_scores_per_strategy[strategy].append(cumulative_totals[strategy])

        # Update cumulative totals with payoffs from the current round
        for strategy_label, payoff in data_by_round.get(server_round, []):
            cumulative_totals[strategy_label] += payoff

    # Append the final cumulative totals after the last round
    for strategy in strategies:
        cumulative_scores_per_strategy[strategy].append(cumulative_totals[strategy])

    # Extend the rounds list to match the length of cumulative scores lists
    extended_rounds = sorted_rounds + [sorted_rounds[-1] + 1]

    # Plot the cumulative total scores over rounds for each strategy
    plt.figure(figsize=(12, 8))
    for strategy, cumulative_scores in cumulative_scores_per_strategy.items():
        plt.plot(extended_rounds, cumulative_scores, label=strategy)

    plt.title("Cumulative Total Scores of Strategies Over Server Rounds")
    plt.xlabel("Server Round")
    plt.ylabel("Cumulative Total Score")
    plt.legend(title="Strategy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
    
def plot_strategy_score_differences_matrix(ipd_scoreboard_dict):
    """
    Plots a confusion matrix showing the score differences (sum_score1 - sum_score2)
    for each strategy pair.

    Each cell displays the difference in total scores between two strategies across all interactions.
    Positive values indicate that the client strategy performed better, negative values indicate that the opponent strategy performed better.
    """
    # Dictionary to store accumulated scores and interaction counts between unique strategy pairs
    # Key: (client_label, opponent_label)
    # Value: [sum_client_scores, sum_opponent_scores]
    interaction_data = {}

    # Iterate through each client to collect scores for unique interactions
    for client_id, rounds in ipd_scoreboard_dict.items():
        for round_data in rounds:
            # round_data format:
            # (server_round, match_id, opponent_id, play, coplay, payoff, ipd_strategy, res_level)
            match_id = round_data[1]
            client_label = f"{round_data[6]} | {round_data[7]}"  # Client's strategy and resource level
            opponent_id = round_data[2]
            client_payoff = round_data[5]  # Client's payoff for this round

            # Ensure the opponent exists and retrieve opponent data for the same match
            if opponent_id in ipd_scoreboard_dict:
                # Find the opponent's data within the same match
                opponent_round = next((r for r in ipd_scoreboard_dict[opponent_id] if r[1] == match_id), None)
                if opponent_round:
                    opponent_label = f"{opponent_round[6]} | {opponent_round[7]}"
                    opponent_payoff = opponent_round[5]  # Opponent's payoff for this round

                    # Create a label pair
                    label_pair = (client_label, opponent_label)

                    # Initialize or update the interaction data
                    if label_pair not in interaction_data:
                        interaction_data[label_pair] = [client_payoff, opponent_payoff]
                    else:
                        interaction_data[label_pair][0] += client_payoff
                        interaction_data[label_pair][1] += opponent_payoff

    # Extract unique labels
    unique_labels = sorted(set(label for pair in interaction_data.keys() for label in pair))

    # Create a DataFrame to store the score differences
    matrix = pd.DataFrame(0.0, index=unique_labels, columns=unique_labels)

    # Fill the matrix with the score differences
    for (client_label, opponent_label), (sum_score1, sum_score2) in interaction_data.items():
        score_difference = sum_score1 - sum_score2
        matrix.at[client_label, opponent_label] = score_difference

    # Plot the confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="coolwarm", center=0, cbar=True)
    plt.title("Strategy Score Differences Matrix (Client Score - Opponent Score)")
    plt.xlabel("Opponent Strategy")
    plt.ylabel("Client Strategy")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    
    
def save_strategy_score_differences_matrix(ipd_scoreboard_dict, plot_directory='plots', filename='strategy_score_differences_matrix.png'):
    """
    Calculates the score differences (sum_score1 - sum_score2) for each strategy pair and saves the plot.

    Parameters:
    - plot_directory (str): The directory where the plot image will be saved.
    - filename (str): The filename for the saved plot image.

    The plot will be saved in the specified directory with the given filename.
    """
    # Ensure the plot directory exists
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    # Dictionary to store accumulated scores between unique strategy pairs
    interaction_data = {}

    # Iterate through each client to collect scores for unique interactions
    for client_id, rounds in ipd_scoreboard_dict.items():
        for round_data in rounds:
            # round_data format:
            # (server_round, match_id, opponent_id, play, coplay, payoff, ipd_strategy, res_level)
            match_id = round_data[1]
            client_label = f"{round_data[6]} | {round_data[7]}"  # Client's strategy and resource level
            opponent_id = round_data[2]
            client_payoff = round_data[5]  # Client's payoff for this round

            # Ensure the opponent exists and retrieve opponent data for the same match
            if opponent_id in ipd_scoreboard_dict:
                # Find the opponent's data within the same match
                opponent_round = next((r for r in ipd_scoreboard_dict[opponent_id] if r[1] == match_id), None)
                if opponent_round:
                    opponent_label = f"{opponent_round[6]} | {opponent_round[7]}"
                    opponent_payoff = opponent_round[5]  # Opponent's payoff for this round

                    # Create a label pair
                    label_pair = (client_label, opponent_label)

                    # Initialize or update the interaction data
                    if label_pair not in interaction_data:
                        interaction_data[label_pair] = [client_payoff, opponent_payoff]
                    else:
                        interaction_data[label_pair][0] += client_payoff
                        interaction_data[label_pair][1] += opponent_payoff

    # Extract unique labels
    unique_labels = sorted(set(label for pair in interaction_data.keys() for label in pair))

    # Create a DataFrame to store the score differences
    matrix = pd.DataFrame(0.0, index=unique_labels, columns=unique_labels)

    # Fill the matrix with the score differences
    for (client_label, opponent_label), (sum_score1, sum_score2) in interaction_data.items():
        score_difference = sum_score1 - sum_score2
        matrix.at[client_label, opponent_label] = score_difference

    # Plot the confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1f",
        cmap="coolwarm",
        center=0,
        cbar=True,
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title("Strategy Score Differences Matrix (Client Score - Opponent Score)")
    plt.xlabel("Opponent Strategy")
    plt.ylabel("Client Strategy")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save the plot to the specified directory with the given filename
    plot_path = os.path.join(plot_directory, filename)
    plt.savefig(plot_path)
    plt.close()  # Close the figure to free memory
    
    
def save_strategy_total_scores_over_rounds(ipd_scoreboard_dict, plot_directory='plots', filename='strategy_total_scores_over_rounds.png'):
    """
    Plots the cumulative total scores obtained by each strategy over the server rounds and saves the plot to a file.

    Parameters:
    - plot_directory (str): The directory where the plot image will be saved.
    - filename (str): The filename for the saved plot image.

    The plot will be saved in the specified directory with the given filename.
    """
    # Ensure the plot directory exists
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    # Collect all unique server rounds and strategies
    all_rounds = set()
    strategies = set()
    data_list = []

    # Gather data from ipd_scoreboard_dict
    for client_id, rounds in ipd_scoreboard_dict.items():
        for round_data in rounds:
            server_round = round_data[0]
            strategy_label = f"{round_data[6]} | {round_data[7]}"  # Strategy name | Resource level
            payoff = round_data[5]  # Payoff

            all_rounds.add(server_round)
            strategies.add(strategy_label)
            data_list.append((server_round, strategy_label, payoff))

    # Sort the server rounds
    sorted_rounds = sorted(all_rounds)

    # Initialize cumulative scores and totals for each strategy
    cumulative_scores_per_strategy = {strategy: [] for strategy in strategies}
    cumulative_totals = {strategy: 0 for strategy in strategies}

    # Group data by server round
    data_by_round = defaultdict(list)
    for server_round, strategy_label, payoff in data_list:
        data_by_round[server_round].append((strategy_label, payoff))

    # Iterate over each server round in order
    for server_round in sorted_rounds:
        # Append current cumulative totals to the lists
        for strategy in strategies:
            cumulative_scores_per_strategy[strategy].append(cumulative_totals[strategy])

        # Update cumulative totals with payoffs from the current round
        for strategy_label, payoff in data_by_round.get(server_round, []):
            cumulative_totals[strategy_label] += payoff

    # Append the final cumulative totals after the last round
    for strategy in strategies:
        cumulative_scores_per_strategy[strategy].append(cumulative_totals[strategy])

    # Extend the rounds list to match the length of cumulative scores lists
    extended_rounds = sorted_rounds + [sorted_rounds[-1] + 1]

    # Plot the cumulative total scores over rounds for each strategy
    plt.figure(figsize=(12, 8))
    for strategy, cumulative_scores in cumulative_scores_per_strategy.items():
        plt.plot(extended_rounds, cumulative_scores, label=strategy)

    plt.title("Cumulative Total Scores of Strategies Over Server Rounds")
    plt.xlabel("Server Round")
    plt.ylabel("Cumulative Total Score")
    plt.legend(title="Strategy")
    plt.grid(True)
    plt.tight_layout()

    # Save the plot to the specified directory with the given filename
    plot_path = os.path.join(plot_directory, filename)
    plt.savefig(plot_path)
    plt.close()  # Close the figure to free memory
