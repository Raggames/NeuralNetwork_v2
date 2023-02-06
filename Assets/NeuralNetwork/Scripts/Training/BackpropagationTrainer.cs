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
        public NetworkBuilder Builder;
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


        [ReadOnly] public float Accuracy;
        [ReadOnly] public int correctRuns;
        [ReadOnly] public int wrongRuns;

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
        public float BestLoss;
        public float CurrentLoss;

        private WaitForSeconds delay;
        private Coroutine ExecutionCoroutine;

        [Header("----RUNTIME----")]
        public double[][] x_datas;
        public double[][] y_datas;

        [Header("Real Time In/Out")]
        public double[] _run_inputs;
        public double[] _run_outputs;
        public double[] _run_test_outputs;

        public void Start()
        {
            if (!AutoStart)
                return;

            Initialize();

            if (Mode == RunningMode.Train)
            {
                PrepareTraining();

                ExecutionCoroutine = StartCoroutine(DoBackPropagationTraining(Epochs));
            }
            else
            {
                PrepareExecution();

                ExecutionCoroutine = StartCoroutine(DoExecuting(Epochs));
            }
        }

        public void Initialize()
        {
            NeuralNetwork = new NeuralNetwork();
            NeuralNetwork.CreateNetwork(this, Builder);
        }

        public void PrepareTraining()
        {
            TrainingSetting.Init();
            InitializeTrainingBestWeightSet(NeuralNetwork);
        }

        public void PrepareExecution()
        {
            TrainingSetting.Init();
            NeuralNetwork.LoadAndSetWeights(LoadDataByName(SaveName));
        }

        private IEnumerator DoExecuting(int runs)
        {
            int count = 0;
            for (int i = 0; i < runs; ++i)
            {
                TrainingSetting.GetNextValues(out _run_inputs, out _run_test_outputs);
                NeuralNetwork.FeedForward(_run_inputs, out _run_outputs);
                ComputeAccuracy(_run_test_outputs, _run_outputs);

                CurrentEpoch++;
                count++;
                yield return null;
            }

        }

        #region BackpropagationTraining

        private IEnumerator DoBackPropagationTraining(int runs)
        {
            double time = 0;
            double time2 = 0;
            int epochcount = 0;

            for (int i = 0; i < runs; ++i)
            {
                ExecuteEpoch();

                CurrentEpoch++;
                epochcount++;

                epochs_per_second = epochcount / Time.realtimeSinceStartup;
                time += Time.deltaTime;
                if (time > max_frame_time)
                {
                    time = 0;
                    yield return null;
                    time2 += Time.deltaTime;
                }
            }

            yield return delay;

            //NeuralNetwork.GetAndSaveWeights();
            SaveBestTrainingWeightSet(NeuralNetwork);
        }

        public virtual void ExecuteEpoch()
        {
            ExecuteFeedForward();

            ComputeLoss(_run_outputs, _run_test_outputs);
            NeuralNetwork.BackPropagate(CurrentLoss, _run_outputs, _run_test_outputs, LearningRate, Momentum, WeightDecay, BiasRate);

            EndEpoch();
        }

        public void EndEpoch()
        {
            ComputeAccuracy(_run_test_outputs, _run_outputs);

            ComputeLearningRateDecay();

            if (Accuracy > Training_Best_Accuracy)
            {
                // Keeping a trace of the set
                MemorizeBestSet(NeuralNetwork, Accuracy);
            }
        }

        public void ExecuteFeedForward()
        {
            TrainingSetting.GetNextValues(out _run_inputs, out _run_test_outputs);
            NeuralNetwork.FeedForward(_run_inputs, out _run_outputs);
        }

        public void ComputeLearningRateDecay()
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

        public void ComputeLoss(double[] outputs = null, double[] testValues = null)
        {
            double lossResult = 0f;
            double[] errors = ComputeError(_run_outputs, testValues);

            switch (LossFunction)
            {
                case LossFunctions.MeanSquarredError:

                    for (int i = 0; i < errors.Length; ++i)
                    {
                        lossResult += Math.Pow(2, errors[i]);
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

            if (lossResult <= BestLoss)
            {
                BestLoss = (float)lossResult;
            }

            CurrentLoss = (float)lossResult;
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
