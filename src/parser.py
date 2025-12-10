import argparse

parser = argparse.ArgumentParser(description='Time-Series Anomaly Detection')
parser.add_argument('--dataset', 
					metavar='-d', 
					type=str, 
					required=False,
					default='synthetic',
                    help="dataset from ['synthetic', 'SMD']")
parser.add_argument('--model', 
					metavar='-m', 
					type=str, 
					required=False,
					default='LSTM_Multivariate',
                    help="model name")

parser.add_argument('--batch_size', 
					metavar='-bsz', 
					type=int, 
					required=False,
					default='128',
                    help="bsz")

parser.add_argument('--total_epochs', 
					metavar='-t_epoch', 
					type=int, 
					required=False,
					default='5',
                    help="epochs")

parser.add_argument('--test', 
					action='store_true', 
					help="test the model")
parser.add_argument('--retrain', 
					action='store_true', 
					help="retrain the model")
parser.add_argument('--less', 
					action='store_true', 
					help="train using less data")
args = parser.parse_args()