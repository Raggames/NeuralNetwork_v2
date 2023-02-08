using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Unity.Collections;
using UnityEngine;

namespace NeuralNetwork
{
    public enum LossFunctions
    {
        MeanSquarredError, // Regression
        MeanAbsoluteError,
        MeanCrossEntropy, // Binary Classification
        HingeLoss, // Binary Classification
        MultiClassCrossEntropy, // Multiclass Classification
    }

    public class BackpropagationTrainer : NeuralNetworkTrainer
    {
        [Header("----PARAMS----")]
        public bool AutoStart = true;

        [Header("----NETWORK----")]
        public ModelBuilder Builder;
        public NeuralNetwork NeuralNetwork;

        // *************************************************************************************
        [Header("----MODE----")]
        public RunningMode Mode;
        public enum RunningMode
        {
            Train,
            Execute,
        }

        [Header("---- SAVING ----")]
        public bool Save;
        public int AutoSaveEveryEpochs = 5000;

        [Header("---- TRAINING PARAMETERS ----")]
        public bool StopOnAccuracyAchieved;
        public float TargetAccuracy = 99.9f; // précision voulue

        // The error the training should achieve, stopping condition
        public float Target_Mean_Error = 0.05f;

       
        /// <summary>
        /// Feedback on computation 'speed'
        /// </summary>
        [ReadOnly] public double epochs_per_second = 0;
        /// <summary>
        /// Avoid unity from freezing on compute large epoch values
        /// </summary>
        [ReadOnly] public float max_frame_time = 2;

        [Header("---- LOSS ----")]
        public LossFunctions LossFunction;
        public double Current_Mean_Error;

        private WaitForSeconds delay;
        private Coroutine ExecutionCoroutine;

        [Header("----RUNTIME----")]
        [ReadOnly] public float Accuracy;
        [ReadOnly] public int correctRuns;
        [ReadOnly] public int wrongRuns;

        public double[][] x_datas;
        public double[][] t_datas;

        public double[] run_inputs;
        public double[] run_outputs;
        public double[] run_test_outputs;

        public void Start()
        {
            if (!AutoStart)
                return;

            Initialize();

            if (Mode == RunningMode.Train)
            {
                PrepareTraining();

                ExecutionCoroutine = StartCoroutine(DoBackPropagationTraining());
            }
            else
            {
                PrepareExecution();

                ExecutionCoroutine = StartCoroutine(DoExecuting(Epochs));
            }
        }

        private void OnGUI()
        {
            if (GUI.Button(new Rect(10, 10, 100, 30), "Train"))
            {
                PrepareTraining();

                ExecutionCoroutine = StartCoroutine(DoBackPropagationTraining());
            }

            if (GUI.Button(new Rect(10, 100, 100, 30), "Load"))
            {
                PrepareExecution();
            }

            if (GUI.Button(new Rect(10, 150, 100, 30), "Test"))
            {
                ExecutionCoroutine = StartCoroutine(DoExecuting(Epochs));
            }

            if (GUI.Button(new Rect(10, 50, 100, 30), "Save"))
            {
                SaveBestTrainingWeightSet(NeuralNetwork);
            }
        }

        public void Initialize()
        {
            NeuralNetwork = new NeuralNetwork();
            NeuralNetwork.CreateNetwork(this, Builder);
            TrainingSetting.Init();
        }

        public void PrepareTraining()
        {
            NeuralNetwork.InitializeWeights();
            InitializeTrainingBestWeightSet(NeuralNetwork);
        }

        public void PrepareExecution()
        {
            NeuralNetwork.LoadAndSetWeights(LoadDataByName(SaveName));
        }

        private IEnumerator DoExecuting(int runs)
        {
            correctRuns = 0;
            wrongRuns = 0;

            int count = 0;

            TrainingSetting.GetTrainDatas(out x_datas, out t_datas);

            int[] sequence_indexes = new int[x_datas.Length];
            for (int i = 0; i < sequence_indexes.Length; ++i)
                sequence_indexes[i] = i;

            Shuffle(sequence_indexes);

            for (int i = 0; i < runs; ++i)
            {
                run_inputs = x_datas[sequence_indexes[i]];
                run_test_outputs = t_datas[sequence_indexes[i]]; 
                
                NeuralNetwork.FeedForward(run_inputs, out run_outputs);
                ComputeAccuracy(run_test_outputs, run_outputs);

                CurrentEpoch++;
                count++;
                yield return null;
            }

        }

        #region BackpropagationTraining

        private IEnumerator DoBackPropagationTraining()
        {
            CurrentEpoch = 0;

            double current_time = 0;

            // Get training datas from the setting
            TrainingSetting.GetTrainDatas(out x_datas, out t_datas);

            // Compute number of iterations 
            // Batchsize shouldn't be 0
            int iterations_per_epoch = x_datas.Length / BatchSize;

            int[] sequence_indexes = new int[x_datas.Length];
            for (int i = 0; i < sequence_indexes.Length; ++i)
                sequence_indexes[i] = i;

            for (int i = 0; i < Epochs; ++i)
            {
                int dataIndex = 0;

                // Shuffle datas each epoch
                Shuffle(sequence_indexes);

                // Going through all data batched, mini-batch or stochastic, depending on BatchSize value
                double mean_error_sum = 0;
                for(int d = 0; d < iterations_per_epoch; ++d)
                {
                    for (int j = 0; j < BatchSize; ++j)
                    {
                        run_inputs = x_datas[sequence_indexes[dataIndex]];
                        run_test_outputs = t_datas[sequence_indexes[dataIndex]];

                        NeuralNetwork.FeedForward(run_inputs, out run_outputs);
                        NeuralNetwork.ComputeGradients(run_test_outputs, run_outputs);

                        ComputeAccuracy(run_test_outputs, run_outputs);

                        mean_error_sum += GetLoss(run_outputs, run_test_outputs);
                        dataIndex++;                                              
                        
                        current_time += Time.deltaTime;
                        if (current_time > max_frame_time)
                        {
                            current_time = 0;
                            yield return null;
                        }
                    }

                    // Computing gradients average over batchsize
                    NeuralNetwork.MeanGradients(BatchSize);
                    // Computing new weights and reseting gradients to 0 for next batch
                    NeuralNetwork.ComputeWeights(LearningRate, Momentum, WeightDecay, BiasRate);
                }

                double last_mean_error = Current_Mean_Error;
                // Computing the mean error
                Current_Mean_Error = mean_error_sum / x_datas.Length;

                if (Current_Mean_Error < last_mean_error)
                {
                    // Keeping a trace of the set
                    MemorizeBestSet(NeuralNetwork, Current_Mean_Error);
                }

                // If under target error, stop
                if (Current_Mean_Error < Target_Mean_Error)
                {
                    break;
                }

                DecayLearningRate();

                CurrentEpoch++;
                epochs_per_second = CurrentEpoch / Time.realtimeSinceStartup;
            }

            //NeuralNetwork.GetAndSaveWeights();
        }

        private static void Shuffle(int[] sequence)
        {
            for (int i = 0; i < sequence.Length; ++i)
            {
                int r = UnityEngine.Random.Range(i, sequence.Length);
                int tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }
        }

        public void ExecuteFeedForward()
        {
            TrainingSetting.GetNextValues(out run_inputs, out run_test_outputs);
            NeuralNetwork.FeedForward(run_inputs, out run_outputs);
        }

        public void DecayLearningRate()
        {
            LearningRate -= LearningRate * LearningRateDecay;
        }

        private double[] ComputeError(double[] runResults, double[] testValues)
        {
            double[] cost = new double[runResults.Length];
            for (int i = 0; i < runResults.Length; ++i)
            {
                cost[i] = testValues[i] - runResults[i];
            }
            return cost;
        }

        public double GetLoss(double[] outputs = null, double[] testValues = null)
        {
            double lossResult = 0f;
            double[] errors = ComputeError(outputs, testValues);

            switch (LossFunction)
            {
                case LossFunctions.MeanSquarredError:

                    for (int i = 0; i < errors.Length; ++i)
                    {
                        lossResult += Math.Pow(errors[i], 2);
                    }

                    lossResult /= errors.Length;
                    break;

                case LossFunctions.MeanAbsoluteError:

                    for (int i = 0; i < errors.Length; ++i)
                    {
                        lossResult += Math.Abs(errors[i]);
                    }

                    lossResult /= errors.Length;
                    break;

                case LossFunctions.MeanCrossEntropy:

                    for (int i = 0; i < outputs.Length; ++i)
                    {
                        lossResult += Math.Log(outputs[i]) * testValues[i];
                    }
                    lossResult = -1.0f * lossResult / outputs.Length;

                    break;

                case LossFunctions.HingeLoss:
                    break;

                case LossFunctions.MultiClassCrossEntropy:
                    break;
            }

            return lossResult;
        }

        public bool ComputeAccuracy(double[] tValues, double[] results)
        {
            bool correct = false;

            if (TrainingSetting.ValidateRun(results, tValues))
            {
                correct = true;
                correctRuns++;
            }
            else
            {
                correct = false;
                wrongRuns++;

            }

            Accuracy = ((float)correctRuns * 1) / (float)(correctRuns + wrongRuns); // ugly 2 - check for divide by zero
            Accuracy *= 100f;

            if (Accuracy >= TargetAccuracy && StopOnAccuracyAchieved)
            {
                if (ExecutionCoroutine != null)
                {
                    StopCoroutine(ExecutionCoroutine);
                    Debug.LogError("Training stopped : goal achieved");
                    SaveBestTrainingWeightSet(NeuralNetwork);
                }
            }

            return correct;
        }

        #endregion

    }
}
