import sys
import os

import simulation as sim
import analysis as ar
import training as tr
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
        print("5: manage trained models")
        print("6: exit")
        choice = input("Select an option [1-6]: ").strip()
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
            print('\nTrain DQN - Select Training Mode:')
            print('  1) Quick Training (basic settings)')
            print('  2) Advanced Training (comprehensive configuration)')
            print('  3) Load and Resume Training')
            print('  4) Back to main menu')
            
            train_choice = input('Select training mode [1-4]: ').strip()
            
            if train_choice == '1':
                # Quick training
                print('\nQuick Training Mode')
                print('  a) DQNSmall')
                print('  b) DQNMedium') 
                print('  c) DQNLarge')
                sel = input('Select model [a-c]: ').strip().lower()
                
                variant = input('Variant (standard/sparse/dense/no_chain) [standard]: ').strip() or 'standard'
                try:
                    episodes = int(input('Training episodes [2000]: ') or '2000')
                except ValueError:
                    episodes = 2000
                
                cls = DQNSmall if sel == 'a' else (DQNMedium if sel == 'b' else DQNLarge)
                print(f'\nStarting quick training: {cls.__name__} for {episodes} episodes...')
                
                try:
                    results = tr.quick_train(cls, episodes=episodes, variant=variant)
                    print(f'\nTraining completed!')
                    print(f'Best win rate achieved: {results["best_win_rate"]:.3f}')
                    print(f'Training data saved to: {results["session_dir"]}')
                except Exception as e:
                    print(f'Training failed: {e}')
                    
            elif train_choice == '2':
                # Advanced training
                print('\nAdvanced Training Configuration')
                print('  a) DQNSmall')
                print('  b) DQNMedium')
                print('  c) DQNLarge')
                sel = input('Select model [a-c]: ').strip().lower()
                cls = DQNSmall if sel == 'a' else (DQNMedium if sel == 'b' else DQNLarge)
                
                # Collect advanced parameters
                variant = input('Variant (standard/sparse/dense/no_chain) [standard]: ').strip() or 'standard'
                
                try:
                    episodes = int(input('Total training episodes [5000]: ') or '5000')
                    eval_interval = int(input('Evaluation interval [250]: ') or '250')
                    checkpoint_interval = int(input('Checkpoint interval [500]: ') or '500')
                    patience = int(input('Early stopping patience [1000]: ') or '1000')
                except ValueError:
                    print('Invalid input, using defaults')
                    episodes, eval_interval, checkpoint_interval, patience = 5000, 250, 500, 1000
                
                # Create advanced configuration
                config = tr.create_training_config(
                    total_episodes=episodes,
                    variant=variant,
                    eval_interval=eval_interval,
                    checkpoint_interval=checkpoint_interval,
                    patience=patience,
                    eval_episodes=50,
                    detailed_logging=True
                )
                
                print(f'\nStarting advanced training: {cls.__name__}...')
                print(f'Episodes: {episodes}, Eval every: {eval_interval}, Checkpoints every: {checkpoint_interval}')
                
                try:
                    trainer = tr.DQNTrainer(config)
                    results = trainer.train(cls)
                    print(f'\nAdvanced training completed!')
                    print(f'Best win rate achieved: {results["best_win_rate"]:.3f}')
                    print(f'Episodes completed: {results["episodes_completed"]}')
                    print(f'Training data saved to: {results["session_dir"]}')
                except Exception as e:
                    print(f'Training failed: {e}')
                    
            elif train_choice == '3':
                # Load and resume (placeholder for now)
                print('\nLoad and Resume Training')
                print('This feature will be available in a future update.')
                print('You can manually load checkpoints using the checkpoint system.')
                
            elif train_choice == '4':
                continue  # Back to main menu
            else:
                print('Invalid choice, returning to main menu.')
        elif choice == '5':
            print('\nManage Trained Models')
            print('  1) List training sessions')
            print('  2) Evaluate trained model')
            print('  3) Compare models')
            print('  4) Back to main menu')
            
            manage_choice = input('Select option [1-4]: ').strip()
            
            if manage_choice == '1':
                # List training sessions
                sessions = tr.list_training_sessions()
                if not sessions:
                    print('No training sessions found.')
                else:
                    print('\nAvailable training sessions:')
                    for i, session in enumerate(sessions, 1):
                        print(f'{i}. {session["name"]}')
                        if 'total_episodes' in session:
                            print(f'   Episodes: {session["total_episodes"]}, Variant: {session["variant"]}')
                        if 'best_score' in session:
                            print(f'   Best score: {session["best_score"]:.3f}, Checkpoints: {session.get("checkpoint_count", 0)}')
                        print()
                        
            elif manage_choice == '2':
                # Evaluate trained model
                sessions = tr.list_training_sessions()
                if not sessions:
                    print('No training sessions found.')
                else:
                    print('\nSelect training session to evaluate:')
                    for i, session in enumerate(sessions, 1):
                        print(f'{i}. {session["name"]}')
                    
                    try:
                        session_idx = int(input('Enter session number: ').strip()) - 1
                        if 0 <= session_idx < len(sessions):
                            session = sessions[session_idx]
                            
                            # Find best checkpoint
                            checkpoint_dir = os.path.join(session['path'], 'checkpoints')
                            best_checkpoint = os.path.join(checkpoint_dir, 'best_checkpoint.pth')
                            
                            if os.path.exists(best_checkpoint):
                                # Determine model type from session name
                                session_name = session['name'].lower()
                                if 'small' in session_name:
                                    agent_class = DQNSmall
                                elif 'medium' in session_name:
                                    agent_class = DQNMedium
                                elif 'large' in session_name:
                                    agent_class = DQNLarge
                                else:
                                    print('Cannot determine model type from session name.')
                                    print('Available: DQNSmall (a), DQNMedium (b), DQNLarge (c)')
                                    model_choice = input('Select model type [a-c]: ').strip().lower()
                                    agent_class = DQNSmall if model_choice == 'a' else (DQNMedium if model_choice == 'b' else DQNLarge)
                                
                                try:
                                    episodes = int(input('Evaluation episodes [100]: ') or '100')
                                except ValueError:
                                    episodes = 100
                                
                                print(f'Evaluating {session["name"]} with {episodes} episodes...')
                                results = tr.evaluate_trained_model(best_checkpoint, agent_class, episodes)
                                
                            else:
                                print('No best checkpoint found for this session.')
                        else:
                            print('Invalid session number.')
                    except ValueError:
                        print('Invalid input.')
                        
            elif manage_choice == '3':
                print('\nModel comparison feature will be available in a future update.')
                
            elif manage_choice == '4':
                continue  # Back to main menu
            else:
                print('Invalid choice, returning to main menu.')
                
        elif choice == '6':
            print('Goodbye')
            sys.exit(0)
        else:
            print('Invalid choice, try again.')


if __name__ == '__main__':
    main()
