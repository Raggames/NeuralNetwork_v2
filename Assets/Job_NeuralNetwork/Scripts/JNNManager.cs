using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Unity.Collections;
using UnityEngine;

namespace Assets.Job_NeuralNetwork.Scripts
{
    public class JNNManager : MonoBehaviour
    {
        public JNN Network;

        public UILineRenderer LearningRateGraph;
        public UILineRenderer LossGraph;

        // *************************************************************************************
        [Header("Training Parameters")] 

        public int Epochs; // trainData lenght
        public int currentEpoch;
        public int BatchSize; // every nbr of runs would you want to compute error

        [Range(0.00001f, 0.5f)] public float LearningRate;
        public double Momentum;

        public double WeightDecay = 0.0001f;

        public enum LossFunctions
        {
            MeanSquarredError, // Regression
            MeanAbsoluteError,
            CrossEntropy, // Binary Classification
            HingeLoss, // Binary Classification
            MultiClassCrossEntropy, // Multiclass Classification
        }
        [Header("Loss Functions")]
        public LossFunctions LossFunction;

        public float TargetLoss = 0.01f; // précision voulue
        public double BestLoss;
        public double CurrentLoss;

        public enum TrainingMode
        {
            Online,
            Batch,
        }
        [Header("Training Mode")]

        public TrainingMode trainingMode;

        private WaitForSeconds delay;
        public float DelayBetweenEpochs = 0.05f;
        private Coroutine TrainingCoroutine;


        // **************************************************************************************
        private double[] errorBatch;

        public double[][] TestingDataX;
        public double[][] TestingDataY;

        [Header("Real Time In/Out")]
        public double[] runInputs;
        public double[] runResults;
        public double[] runWantedOutpus;
    
        public List<LearningData> TrainingDatas = new List<LearningData>();

        public void Start()
        {
            delay = new WaitForSeconds(DelayBetweenEpochs);

            Network.CreateNetwork(this);

            TestingDataX = new double[20][];
            TestingDataY = new double[20][];
            for(int i = 0; i < TestingDataX.Length; ++i)
            {
                TestingDataX[i] = new double[2];
                TestingDataY[i] = new double[1];
                double test1 = UnityEngine.Random.Range(0.01f, 1f);
                double test2 = UnityEngine.Random.Range(0.01f, 1f);
                double test3 = 0.25f;
                TestingDataX[i][0] = test1;
                TestingDataX[i][1] = test2;
                TestingDataY[i][0] = test3;

            }

            BestLoss = 1;

            RunTest();
        }

        public void RunTest()
        {
            TrainingCoroutine = StartCoroutine(RunDelayed(Epochs));
        }

        public void Train()
        {

        }

        public void Execute()
        {

        }



        private IEnumerator RunDelayed(int runs)
        {
            int countBatch = 0;
            errorBatch = new double[Network.OutputLayer.NeuronsCount];

            for (int i = 0; i < runs; ++i)
            {
                int index = UnityEngine.Random.Range(0, 19);
                runInputs = TestingDataX[index];
                runWantedOutpus = TestingDataY[index];
                NativeArray<double> inputTest = Network.ToNativeArray(TestingDataX[index]);
                double[] testValues = TestingDataY[index]; // testValues for each output neuron.
               
                Network.ExecuteForward(inputTest, out runResults);
                double[] errors = ComputeError(runResults, testValues);

                if (trainingMode == TrainingMode.Batch)
                {
                    countBatch++;
                             
                    for(int b = 0; b < errorBatch.Length; ++b)
                    {
                        errorBatch[b] += errors[b];
                    }

                    if (countBatch > BatchSize)
                    {
                        for (int b = 0; b < errorBatch.Length; ++b)
                        {
                            errorBatch[b] /= errorBatch.Length;
                        }
                        GetLearningData(errorBatch); // <= sur la moyenne des erreurs

                        // Backpropagate
                        Network.BackPropagate(errorBatch, LearningRate);

                        countBatch = 0;

                        for (int b = 0; b < errorBatch.Length; ++b)
                        {
                            errorBatch[b] = 0;
                        }
                    }
                }
                else
                {
                    GetLearningData(errors);
                    Network.BackPropagate(errors, LearningRate);
                }


                currentEpoch++;
                yield return delay;

            }
        }

        private double[] ComputeError(double[] runResults, double[] testValues)
        {
            double[] cost = new double[runResults.Length];
            for (int i = 0; i < runResults.Length; ++i)
            {
                cost[i] = runResults[i] - testValues[i];
            }
            return cost;
        }

        private double ComputeLoss(double[] errors)
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

                case LossFunctions.CrossEntropy:
                    break;

                case LossFunctions.HingeLoss:
                    break;

                case LossFunctions.MultiClassCrossEntropy:
                    break;
            }

            return lossResult;
        }

        private void GetLearningData(double[] errors)
        {
            CurrentLoss = ComputeLoss(errors);

            LearningData data = new LearningData
            {
                Epoch = currentEpoch,
                LearningRate = LearningRate,
                BestLoss = BestLoss,
                ActualLoss = CurrentLoss,
            };
            TrainingDatas.Add(data);

            LearningRateGraph.points.Add(new Vector2(currentEpoch, LearningRate*100));
            LossGraph.points.Add(new Vector2(currentEpoch, (float)(CurrentLoss*100)));

            if(CurrentLoss <= BestLoss)
            {
                BestLoss = CurrentLoss;
            }

            if (BestLoss <= TargetLoss)
            {
                StopCoroutine(TrainingCoroutine);
                Debug.LogError("Training stopped : goal achieved");

            }
        }

        public struct LearningData
        {
            public int Epoch;
            public float LearningRate;
            public double BestLoss;
            public double ActualLoss;
        }
    }
}
