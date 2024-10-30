using System;
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

    public abstract class NeuralNetworkTrainer : MonoBehaviour
    {
        [Header("---- TRAINING SETTING")]
        /// <summary>
        /// Contains the data set and the function to evaluate the accuracy of the network while training
        /// </summary>
        public TrainingSettingBase TrainingSetting;
        [Header("---- SAVE ----")]
        public string SaveName = "";

        [Header("---- PARAMETERS ----")]
        public LossFunctions LossFunction;

        /// <summary>
        /// The range of the weigths at initialization
        /// </summary>
        public Vector2 InitialWeightRange = new Vector2(-.1f, .1f);
        /// <summary>
        /// The random seed for initializing weigths
        /// </summary>
        public int InitialWeightSeed = 0;
        /// <summary>
        /// The multiplier for tweaking weights each epoch/batch
        /// </summary>
        [Range(0.00001f, 2f)] public float LearningRate = .3f;

        /// <summary>
        /// Momentum is a ratio that allows to add to the new computed weights a part of the vector of the previous iteration.
        /// It allows the learning to be faster by decreasing the fluctuations of the process.
        /// </summary>
        public float Momentum = 0.01f;

        /// <summary>
        /// The learning rate for the biases
        /// </summary>
        public float BiasRate = 1;

        /// <summary>
        /// Control the decreasing of the learning rate over epochs
        /// </summary>
        public float LearningRateDecay = 0.0001f;

        /// <summary>
        /// Decreases all the weigths of the network over epochs (should be a very small amount, it can totally cancel the learning if to close to the learning rate.
        /// </summary>
        public float WeightDecay = 0.0001f;

        /// <summary>
        /// Number of iterations of the learning process
        /// </summary>
        public int Epochs = 1;
        /// <summary>
        /// [Not yet implemented] Batch are a way to accumulate iterations before updating weights (accumulating gradients or error)
        /// </summary>
        public int BatchSize = 1;

        [Header("---- LOSS ----")]
        public float Target_Mean_Error = 0.05f;
        public double Current_Mean_Error;

        [Header("---- RUNTIME ----")]
        [ReadOnly] public float Accuracy;
        [ReadOnly] public int correctRuns;
        [ReadOnly] public int wrongRuns;
        [ReadOnly] public int CurrentEpoch;
        [ReadOnly] public double Training_Best_Mean_Error;

        protected double[][,] Training_Best_Weigths;
        protected double[][] Training_Best_Biases;

        protected void InitializeTrainingBestWeightSet(NeuralNetwork neuralNetwork)
        {
            // Each epoch, we keep a trace of the best set
            // If the next one doesn't find a best set, we will retry from this one
            // If the retry iterations are over a threshold, the learning will stop and keed this set as the best
            Training_Best_Weigths = new double[neuralNetwork.DenseLayers.Count][,];
            Training_Best_Biases = new double[neuralNetwork.DenseLayers.Count][];

            for (int i = 0; i < neuralNetwork.DenseLayers.Count; ++i)
            {
                Training_Best_Weigths[i] = NeuralNetworkMathHelper.MakeMatrix(neuralNetwork.DenseLayers[i].Weights.GetLength(0), neuralNetwork.DenseLayers[i].Weights.GetLength(1));
                Training_Best_Biases[i] = new double[neuralNetwork.DenseLayers[i].Biases.Length];
            }
        }

        protected void MemorizeBestSet(NeuralNetwork bestSet, double mean_error)
        {
            Training_Best_Mean_Error = mean_error;

            for (int l = 0; l < bestSet.DenseLayers.Count; ++l)
            {
                for (int i = 0; i < bestSet.DenseLayers[l].Weights.GetLength(0); ++i)
                {
                    for (int k = 0; k < bestSet.DenseLayers[l].Weights.GetLength(1); ++k)
                    {
                        Training_Best_Weigths[l][i, k] = bestSet.DenseLayers[l].Weights[i, k];
                    }
                }

                for (int i = 0; i < bestSet.DenseLayers[l].Biases.Length; ++i)
                {
                    Training_Best_Biases[l][i] = bestSet.DenseLayers[l].Biases[i];
                }
            }
        }

        protected void SetWeightsFromBest(NeuralNetwork network)
        {
            for (int l = 0; l < network.DenseLayers.Count; ++l)
            {
                for (int i = 0; i < network.DenseLayers[l].Weights.GetLength(0); ++i)
                {
                    for (int k = 0; k < network.DenseLayers[l].Weights.GetLength(1); ++k)
                    {
                        network.DenseLayers[l].Weights[i, k] = Training_Best_Weigths[l][i, k];
                    }
                }

                for (int i = 0; i < network.DenseLayers[l].Biases.Length; ++i)
                {
                    network.DenseLayers[l].Biases[i] = Training_Best_Biases[l][i];
                }
            }
        }

        /// <summary>
        /// Pass any network as reference, its weight will be reset to the best training values and saved
        /// </summary>
        /// <param name="referent"></param>
        protected void SaveBestTrainingWeightSet(NeuralNetwork referent)
        {
            SetWeightsFromBest(referent);
            SaveCurrentModel(referent);
        }

        protected void SaveCurrentModel(NeuralNetwork referent)
        {
            double[] weights = referent.GetWeights();

            string version = TrainingSetting.name + "_" + DateTime.Now.Hour + "_" + DateTime.Now.Minute + "_" + DateTime.Now.Second + "_" + referent.ArchitectureString() + "_BestSet";

            NetworkData data = new NetworkData
            {
                Version = version,
                dnaSave = weights,
                learningRate = LearningRate,

                momentum = Momentum,
                weightDecay = WeightDecay,
                accuracy = (float)Training_Best_Mean_Error,
            };

            NetworkDataSerializer.Save(data, data.Version);
        }

        protected NetworkData LoadDataByName(string fileName)
        {
            NetworkData loadedData = new NetworkData();
            loadedData = NetworkDataSerializer.Load(loadedData, fileName);
            return loadedData;
        }

        protected static void Shuffle(int[] sequence)
        {
            for (int i = 0; i < sequence.Length; ++i)
            {
                int r = UnityEngine.Random.Range(i, sequence.Length);
                int tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }
        }


        /// <summary>
        /// Difference between predicted outputs and desired output values
        /// </summary>
        /// <param name="runResults"></param>
        /// <param name="testValues"></param>
        /// <returns></returns>
        protected double[] ComputeError(double[] runResults, double[] testValues)
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

            return correct;
        }

    }
}
