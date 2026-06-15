""""""  		  	   		 	   			  		 			     			  	 
"""  		  	   		 	   			  		 			     			  	 
Test a learner.  (c) 2015 Tucker Balch  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	   			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 	   			  		 			     			  	 
All Rights Reserved  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Template code for CS 4646/7646  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	   			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 	   			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 	   			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 	   			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	   			  		 			     			  	 
or edited.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		 	   			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 	   			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	   			  		 			     			  	 
GT honor code violation.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
-----do not edit anything above this line---  		  	   		 	   			  		 			     			  	 
"""  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import math  		  	   		 	   			  		 			     			  	 
import sys  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import numpy as np  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import LinRegLearner as lrl
import matplotlib.pyplot as plt
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    full_data = np.genfromtxt(sys.argv[1], delimiter=',', skip_header=1)
    data = full_data[:, 1:]

    # compute how much of the data is training and testing
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    train_x = data[:train_rows, 0:-1]
    train_y = data[:train_rows, -1]
    test_x = data[train_rows:, 0:-1]
    test_y = data[train_rows:, -1]

    print(f"{test_x.shape}")
    print(f"{test_y.shape}")

    # create a learner and train it
    learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner
    learner.add_evidence(train_x, train_y)  # train it
    print(learner.author())

    # evaluate in sample
    pred_y = learner.query(train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    print()
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0,1]}")

    # evaluate out of sample
    pred_y = learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0,1]}")

    # Experiment 1

    in_sample_rmse = []
    out_sample_rmse = []
    leaf_sizes = range(1, 25)

    for leaf_size in leaf_sizes:
        dt_learner = dt.DTLearner(leaf_size=leaf_size)
        dt_learner.add_evidence(train_x, train_y)

        # In-sample prediction
        predictions_in = dt_learner.query(train_x)
        rmse_in = np.sqrt(np.mean((train_y - predictions_in) ** 2))
        in_sample_rmse.append(rmse_in)

        # Out-of-sample prediction
        predictions_out = dt_learner.query(test_x)
        rmse_out = np.sqrt(np.mean((test_y - predictions_out) ** 2))
        out_sample_rmse.append(rmse_out)

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(leaf_sizes, in_sample_rmse, label='In-Sample RMSE')
    plt.plot(leaf_sizes, out_sample_rmse, label='Out-of-Sample RMSE')
    plt.title('Decision Tree Learner: In-Sample vs Out-of-Sample RMSE')
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig('Experiment1_Overfitting_Analysis.png')
    plt.close()

    # Experiment 2

    bags_rmse_in_sample = []
    bags_rmse_out_sample = []

    for leaf_size in leaf_sizes:
        bag_learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": leaf_size}, bags=10, boost=False)
        bag_learner.add_evidence(train_x, train_y)

        # In-sample evaluation
        pred_in = bag_learner.query(train_x)
        rmse_in = np.sqrt(np.mean((train_y - pred_in) ** 2))
        bags_rmse_in_sample.append(rmse_in)

        # Out-of-sample evaluation
        pred_out = bag_learner.query(test_x)
        rmse_out = np.sqrt(np.mean((test_y - pred_out) ** 2))
        bags_rmse_out_sample.append(rmse_out)

    # Plotting the bagging effects
    plt.figure(figsize=(10, 6))
    plt.plot(leaf_sizes, bags_rmse_in_sample, label='In-Sample RMSE with Bagging')
    plt.plot(leaf_sizes, bags_rmse_out_sample, label='Out-of-Sample RMSE with Bagging')
    plt.title('Bagging Effect on RMSE by Leaf Size')
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig('Experiment2_Bagging_Effect.png')
    plt.close()

    # Experiment 3

    def mean_absolute_error_manual(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))


    def r2_score_manual(y_true, y_pred):
        sst = np.sum((y_true - np.mean(y_true)) ** 2)
        ssr = np.sum((y_true - y_pred) ** 2)
        if sst < 1e-8:
            return 1.0 if ssr < 1e-8 else 0.0
        return 1 - (ssr / sst)

    mae_dt = []
    mae_rt = []
    r2_dt = []
    r2_rt = []

    leaf_sizes = range(1, 21)

    for leaf_size in leaf_sizes:
        # Initialize and train DTLearner
        dt_learner = dt.DTLearner(leaf_size=leaf_size)
        dt_learner.add_evidence(train_x, train_y)
        predictions_dt = dt_learner.query(train_x)

        # Initialize and train RTLearner
        rt_learner = rt.RTLearner(leaf_size=leaf_size)
        rt_learner.add_evidence(train_x, train_y)
        predictions_rt = rt_learner.query(train_x)

        # Calculate MAE manually
        mae_dt.append(mean_absolute_error_manual(train_y, predictions_dt))
        mae_rt.append(mean_absolute_error_manual(train_y, predictions_rt))

        # Calculate R^2 manually
        r2_dt.append(r2_score_manual(train_y, predictions_dt))
        r2_rt.append(r2_score_manual(train_y, predictions_rt))

    # Plot MAE comparison
    plt.figure(figsize=(10, 6))
    plt.plot(leaf_sizes, mae_dt, label='DTLearner MAE')
    plt.plot(leaf_sizes, mae_rt, label='RTLearner MAE')
    plt.title('MAE Comparison: DTLearner vs RTLearner')
    plt.xlabel('Leaf Size')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.legend()
    plt.savefig('Experiment3_MAE_Comparison.png')
    plt.close()

    # Plot R^2 comparison
    plt.figure(figsize=(10, 6))
    plt.plot(leaf_sizes, r2_dt, label='DTLearner R^2')
    plt.plot(leaf_sizes, r2_rt, label='RTLearner R^2')
    plt.title('R^2 Score Comparison: DTLearner vs RTLearner')
    plt.xlabel('Leaf Size')
    plt.ylabel('Coefficient of Determination (R^2)')
    plt.legend()
    plt.savefig('Experiment3_R2_Comparison.png')
    plt.close()





