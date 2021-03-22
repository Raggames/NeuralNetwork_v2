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
        [Header("Data Manager")]
        public JNNDataManager DataManager;

        [Header("Convolutionnal And Pooling Layers")]
        public List<JNNCPLayer> CPLayers = new List<JNNCPLayer>(); // Convolve and Pool

        [Header("Feed Forward Network")]
        public JNNFeedForward FFNetwork;

        [Header("Graph Rendering")]
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
            MeanCrossEntropy, // Binary Classification
            HingeLoss, // Binary Classification
            MultiClassCrossEntropy, // Multiclass Classification
        }
        [Header("Loss Functions")]
        public LossFunctions LossFunction;

        public float TargetLoss = 0.01f; // précision voulue
        public double BestLoss;
        public double CurrentLoss;

      
        private WaitForSeconds delay;
        public float DelayBetweenEpochs = 0.05f;
        private Coroutine TrainingCoroutine;


        [Header("Learning Rate Decay")]
        public LRDecayMode LearningRateDecayMode;
        public enum LRDecayMode
        {
            None,
            Decay,
            Exponential,
            Step,
            BatchStep,

        }
        public float DecayRate = 0.95f;
        public int DecayStep;

        // **************************************************************************************
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

            FFNetwork.CreateNetwork(this);
            DataManager.Init();

            TestingDataX = new double[150][];
            TestingDataY = new double[150][];

            for (int i = 0; i < TestingDataX.Length; ++i)
            {
                TestingDataX[i] = new double[4];
                TestingDataY[i] = new double[3];

                double[] data = DataManager.GetDataEntry(i);

                for(int j = 0; j < 4; ++j)
                {
                    TestingDataX[i][j] = data[j];
                }

                for(int k = 0; k < 3; ++k)
                {
                    TestingDataY[i][k] = data[4+k];
                }
            }

            BestLoss = 10;

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

            // ********************************* Convolve and Pool

            // ********************************* Flattenning 
            for (int i = 0; i < runs; ++i)
            {
                int index = UnityEngine.Random.Range(0, 149);
                runInputs = TestingDataX[index];

                runWantedOutpus = TestingDataY[index];
                             
                FFNetwork.ComputeFeedForward(TestingDataX[index], out runResults);

                double[] errors = ComputeError(runResults, TestingDataY[index]);

                GetLearningData(errors, runResults, TestingDataY[index]);
                FFNetwork.BackPropagate(errors, LearningRate);

                currentEpoch++;

                if(LearningRateDecayMode != LRDecayMode.None)
                {
                    ComputeLearningRateDecay();
                }

                yield return delay;
            }
        }

        public void ComputeLearningRateDecay()
        {
            switch (LearningRateDecayMode)
            {
                case LRDecayMode.Decay:
                    LearningRate -= LearningRate * DecayRate;

                    break;
                case LRDecayMode.Exponential:
                    LearningRate = (float)(DecayRate * Math.Exp(currentEpoch) * LearningRate);

                    break;
                case LRDecayMode.Step:
                    LearningRate = (float)(DecayRate / Math.Sqrt((double)currentEpoch)) * LearningRate;

                    break;
                case LRDecayMode.BatchStep:
                    LearningRate = (float)(DecayRate / Math.Sqrt((double)(currentEpoch / BatchSize))) * LearningRate;

                    break;
            }
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

        private double ComputeLoss(double[] errors, double[] outputs = null, double[] testValues = null)
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

                    for(int i = 0; i < outputs.Length; ++i)
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

        private void GetLearningData(double[] errors, double[] outputs, double[] testValues)
        {
            CurrentLoss = ComputeLoss(errors, outputs, testValues);
/*
            LearningData data = new LearningData
            {
                Epoch = currentEpoch,
                LearningRate = LearningRate,
                BestLoss = BestLoss,
                ActualLoss = CurrentLoss,
            };
            TrainingDatas.Add(data);*/

            LearningRateGraph.points.Add(new Vector2(currentEpoch, LearningRate*100));
            LossGraph.points.Add(new Vector2(currentEpoch, (float)(CurrentLoss*100)));

            if(CurrentLoss <= BestLoss)
            {
                BestLoss = CurrentLoss;
            }

            if (BestLoss <= TargetLoss)
            {
              
              //  StopCoroutine(TrainingCoroutine);
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
