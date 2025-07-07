
"""
Main script for training and testing a DQN agent on the LunarLander-v3 environment.

Author: AF
Created: 28-06-2022

Description:
    - Offers a menu interface for training a new model or running a demo.
    - Integrates with DQNAgent, model, and HParams modules.
    - Handles interactive plotting and model checkpointing.

"""

import experiments

def show_menu():
    print("Main Menu:")
    print("1. Train and test a new model")
    print("2. Run a demo on existing trained model")
    print("3. Exit")



def main():
    global new_model_path
    while True:
        show_menu()
        choice = input("Enter your choice: ").strip()
        if choice == "1":
            experiments.run_new_experiment()


        elif choice == "2":
            print("You chose to run a pre-existing demo. This demo is based on  a model using 128 neurons and 1 hidden layer.")
            experiments.demo()

        elif choice == "3":
            print("Exiting...")
            experiments.cleanup()

            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")



if __name__ == "__main__":
    main()
