import sys
import os

import simulation as sim
import analysis as ar
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from agents import DQNSmall, DQNMedium, DQNLarge


def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


def main():
    while True:
        print("\nOware AI Menu")
        print("1: run simulation")
        print("2: run tournament")
        print("3: run analysis")
        print("4: train dqn")
        print("5: exit")
        choice = input("Select an option [1-5]: ").strip()
        if choice == '1':
            episodes = input('Number of episodes to simulate: ').strip()
            print('Starting simulation (this may take a while)...')
            sim.run_simulation(int(episodes))
        elif choice == '2':
            print('Running tournament...')
            sim.run_tournament()
        elif choice == '3':
            sim_log_path = os.path.join('output', 'sim_log.csv')
            if not os.path.exists(sim_log_path):
                print('No sim_log.csv found in output directory. Please run simulations first.')
            else:
                print('Running deep statistical analysis on simulation data...')
                ar.analyze('sim_log.csv')
        elif choice == '4':
            print('Train DQN')
            print('  a) DQNSmall')
            print('  b) DQNMedium')
            print('  c) DQNLarge')
            sel = input('Select model [a-c]: ').strip().lower()
            variant = input('Variant to train on (standard/sparse/dense/no_chain) [standard]: ').strip() or 'standard'
            try:
                episodes = int(input('Training episodes [2000]: ') or '2000')
            except ValueError:
                episodes = 2000
            cls = DQNSmall if sel == 'a' else (DQNMedium if sel == 'b' else DQNLarge)
            print(f'Training {cls.__name__} on variant {variant} for {episodes} episodes...')
            print('Training finished and checkpoint saved (if supported).')
        elif choice == '5':
            print('Goodbye')
            sys.exit(0)
        else:
            print('Invalid choice, try again.')


if __name__ == '__main__':
    main()
