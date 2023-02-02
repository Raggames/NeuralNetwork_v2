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
        [Header("Data Manager")]
        public FlowerClassification DataManager;

        [Header("Feed Forward Network")]
        public NetworkBuilder Builder;
        public NeuralNetwork NeuralNetwork;

        // *************************************************************************************
        [Header("RunMode")]
        public RunningMode Mode;
        public enum RunningMode
        {
            Train,
            Execute,
        }
        [Header("Save Management")]
        public bool Save;
        public int AutoSaveEveryEpochs = 5000;

        [Header("Training Parameters")]
        public bool AutomaticStop;

        public float Momentum = 0.01f;
        public float WeightDecay = 0.0001f;
        public float BiasRate = 1;

        public double Epochs_Per_Second = 0;
        public float max_frame_time = 2;

        [Header("Loss and Accuracy Parameters")]
        public LossFunctions LossFunction;

        public float TargetAccuracy = 99.9f; // précision voulue
        public float Accuracy;

        public int correctRuns;
        public int wrongRuns;

        public float BestLoss;
        public float CurrentLoss;

        private WaitForSeconds delay;
        public float DelayBetweenEpochs = 0.05f;
        private Coroutine ExecutionCoroutine;

        [Header("Learning Rate Decay")]

        public float DecayRate = 0.95f;
        public int DecayStep;

        // **************************************************************************************
        public double[][] TestingDataX;
        public double[][] TestingDataY;

        [Header("Real Time In/Out")]
        public double[] runInputs;
        public double[] _run_outputs;
        public double[] _run_test_outputs;

        public void Start()
        {
            delay = new WaitForSeconds(DelayBetweenEpochs);

            NeuralNetwork = new NeuralNetwork();
            NeuralNetwork.CreateNetwork(this, Builder);

            InitTraining();

            if (Mode == RunningMode.Train)
            {
                Train();

            }
            else
            {
                Execute();
            }
        }

        public void Train()
        {
            ExecutionCoroutine = StartCoroutine(DoBackPropagationTraining(Epochs));
        }

        public void Execute()
        {
            NeuralNetwork.LoadAndSetWeights();
            ExecutionCoroutine = StartCoroutine(DoExecuting(Epochs));
        }

        private IEnumerator DoExecuting(int runs)
        {
            // ********************************* Flattenning 
            int count = 0;
            for (int i = 0; i < runs; ++i)
            {
                int index = UnityEngine.Random.Range(0, TestingDataX.Length); // i % 149;
                runInputs = TestingDataX[index];

                _run_test_outputs = TestingDataY[index];

                NeuralNetwork.ComputeFeedForward(TestingDataX[index], out _run_outputs);
                //FFNetwork.JobComputeFeedForward(TestingDataX[index], out runResults);

                ComputeAccuracy(TestingDataY[index], _run_outputs);

                CurrentEpoch++;
                count++;

                yield return delay;
            }
        }

        #region BackpropagationTraining
        /*private void InitTraining()
        {
            DataManager.Init();

            TestingDataX = new double[150][];
            TestingDataY = new double[150][];

            for (int i = 0; i < TestingDataX.Length; ++i)
            {
                TestingDataX[i] = new double[4];
                TestingDataY[i] = new double[3];

                double[] data = DataManager.GetDataEntry(i);

                for (int j = 0; j < 4; ++j)
                {
                    TestingDataX[i][j] = data[j];
                }

                for (int k = 0; k < 3; ++k)
                {
                    TestingDataY[i][k] = data[4 + k];
                }
            }

            BestLoss = 10;
        }*/

        private void InitTraining()
        {
            TestingDataX = new double[15000][];
            TestingDataY = new double[15000][];

            for (int i = 0; i < TestingDataX.Length; ++i)
            {
                TestingDataX[i] = new double[4];
                TestingDataY[i] = new double[1];

                //double[] data = DataManager.GetDataEntry(i);

                TestingDataX[i][0] = UnityEngine.Random.Range(-1f, 1f);
                TestingDataX[i][1] = UnityEngine.Random.Range(-1f, 1f);
                TestingDataX[i][2] = UnityEngine.Random.Range(-1f, 1f);
                TestingDataX[i][3] = UnityEngine.Random.Range(-1f, 1f);

                double sum = TestingDataX[i][0] + TestingDataX[i][1] + TestingDataX[i][2] + TestingDataX[i][3];
                TestingDataY[i][0] = sum > 0 ? 1 : 0;
            }

            BestLoss = 10;
        }


        private IEnumerator DoBackPropagationTraining(int runs)
        {
            // ********************************* Convolve and Pool

            // ********************************* Flattenning 
            int count = 0;
            double time = 0;
            double time2 = 0;
            int epochcount = 0;

            for (int i = 0; i < runs; ++i)
            {
                int index = UnityEngine.Random.Range(0, 149); // i % 149;
                runInputs = TestingDataX[index];

                _run_test_outputs = TestingDataY[index];

                NeuralNetwork.ComputeFeedForward(TestingDataX[index], out _run_outputs);
                //FFNetwork.JobComputeFeedForward(TestingDataX[index], out runResults);

                double[] errors = ComputeError(_run_outputs, TestingDataY[index]);
                CurrentLoss = ComputeLoss(errors, _run_outputs, _run_test_outputs);


                if (CurrentLoss <= BestLoss)
                {
                    BestLoss = CurrentLoss;
                }

                ComputeAccuracy(TestingDataY[index], _run_outputs);

                NeuralNetwork.BackPropagate(_run_outputs, _run_test_outputs, LearningRate, Momentum, WeightDecay, BiasRate);

                CurrentEpoch++;
                epochcount++;

                ComputeLearningRateDecay();

                Epochs_Per_Second = epochcount / Time.realtimeSinceStartup;
                time += Time.deltaTime;
                if (time > max_frame_time)
                {
                    time = 0;
                    yield return null;
                    time2 += Time.deltaTime;
                }
            }

            yield return delay;

            NeuralNetwork.GetAndSaveWeights();
        }

        public void ComputeLearningRateDecay()
        {
            LearningRate -= LearningRate * DecayRate;
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

        public float ComputeLoss(double[] errors, double[] outputs = null, double[] testValues = null)
        {
            double lossResult = 0f;

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

            return (float)lossResult;
        }

        private void ComputeAccuracy(double[] tValues, double[] results)
        {
            /*int index = JNNMath.MaxIndex(results);
            int tMaxIndex = JNNMath.MaxIndex(tValues);
            if (index.Equals(tMaxIndex))
            {
                correctRuns++;
            }
            else
            {
                wrongRuns++;
            }*/

            if (Mathf.Abs((float)tValues[0] - (float)results[0]) < .1f)
            {
                correctRuns++;
            }
            else
            {
                wrongRuns++;
            }

            Accuracy = ((float)correctRuns * 1) / (float)(correctRuns + wrongRuns); // ugly 2 - check for divide by zero
            Accuracy *= 100f;

            if (Accuracy >= TargetAccuracy && AutomaticStop)
            {
                if(ExecutionCoroutine != null)
                {
                    StopCoroutine(ExecutionCoroutine);
                    Debug.LogError("Training stopped : goal achieved");
                }             
            }
        }

        #endregion
        
        public struct LearningData
        {
            public int Epoch;
            public float LearningRate;
            public double BestLoss;
            public double ActualLoss;
        }
    }
}
